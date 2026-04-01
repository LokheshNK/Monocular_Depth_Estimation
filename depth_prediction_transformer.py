"""
depth_prediction_transformer.py
─────────────────────────────────────────────────────────────
DepthPredictionTransformer  —  3-Stage Multi-ViT Architecture

ViT is applied THREE times at different scales:
  Stage 1 (EncoderViT)   : on image patches  → global scene layout
  Stage 2 (MidViT)       : on encoder output → depth relationships
  Stage 3 (DecoderViT)   : on decoder output → boundary refinement

MEMORY FIX vs previous version
───────────────────────────────
Problem: Stages 2 & 3 flattened the full spatial map into tokens.
  Stage 2: 32×104 = 3328 tokens → attention matrix 3328×3328 = 11M values
  Stage 3: 64×208 = 13312 tokens → 177M values  → OOM on 6GB GPU

Solution: Pool to fixed small grid BEFORE attention, attend there,
  then upsample back.
  Stage 2: pool to 16×16 = 256 tokens → 256×256 = 65K values  ✓
  Stage 3: pool to 8×8   =  64 tokens →  64×64  = 4K values   ✓

This is identical in spirit to "pooled attention" used in
PoolFormer / PVT and keeps global context while using <1% of the
original memory for attention at Stages 2 and 3.
─────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_model import PatchEmbedding, TransformerBlock


# ══════════════════════════════════════════════════════════
# STAGE 1 COMPONENTS — Encoder ViT
# ══════════════════════════════════════════════════════════

class EncoderViT(nn.Module):
    """
    Full ViT encoder — applied to raw image patches.
    Taps intermediate token maps at 4 depths for multi-scale features.
    """
    def __init__(self, patch_size=16, embed_dim=384, num_heads=6,
                 num_layers=12, dropout=0.1, max_patches=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, patch_size, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 1 + max_patches, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        L = num_layers
        self.hooks = [L//4 - 1, L//2 - 1, L*3//4 - 1, L - 1]

    def _interp_pos(self, N):
        pc = self.pos_embed[:, :1, :]
        pp = self.pos_embed[:, 1:, :]
        if N == pp.shape[1]:
            return self.pos_embed
        pp = F.interpolate(pp.transpose(1,2), N,
                           mode='linear', align_corners=False).transpose(1,2)
        return torch.cat([pc, pp], dim=1)

    def forward(self, x):
        B = x.shape[0]
        tokens, Hp, Wp = self.patch_embed(x)
        N = tokens.shape[1]
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self._interp_pos(N))

        intermediates = {}
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in self.hooks:
                intermediates[i] = tokens[:, 1:, :]

        tokens = self.norm(tokens)
        intermediates[self.hooks[-1]] = tokens[:, 1:, :]
        return intermediates, Hp, Wp


class Reassemble(nn.Module):
    """Tokens [B,N,D] → 2D feature map [B,C,Hp*scale,Wp*scale]."""
    def __init__(self, embed_dim, out_channels, scale=1):
        super().__init__()
        self.proj    = nn.Linear(embed_dim, out_channels)
        self.spatial = (nn.ConvTranspose2d(out_channels, out_channels,
                                           kernel_size=scale, stride=scale)
                        if scale > 1 else nn.Identity())
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, tokens, Hp, Wp):
        x = self.proj(tokens).transpose(1, 2).reshape(-1, self.proj.out_features, Hp, Wp)
        return F.gelu(self.norm(self.spatial(x)))


class FusionBlock(nn.Module):
    """RefineNet-style fusion block with ×2 upsample."""
    def __init__(self, channels):
        super().__init__()
        def rb():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(1, channels), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1))
        self.r1 = rb(); self.r2 = rb()

    def forward(self, x1, x2=None):
        x1 = x1 + self.r1(x1)
        if x2 is not None:
            if x1.shape[2:] != x2.shape[2:]:
                x2 = F.interpolate(x2, x1.shape[2:], mode='bilinear', align_corners=True)
            x1 = x1 + x2
        x1 = x1 + self.r2(x1)
        return F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)


# ══════════════════════════════════════════════════════════
# POOLED ATTENTION BLOCK
# ══════════════════════════════════════════════════════════

class PooledAttentionViT(nn.Module):
    """
    Memory-safe ViT block for high-resolution feature maps.

    PROBLEM with naive approach:
        Flattening a 32×104 map gives 3328 tokens.
        Attention matrix = 3328×3328 × batch × heads = OOM on 6GB GPU.

    SOLUTION — Pool → Attend → Upsample:
        1. Spatially pool feature map down to pool_size×pool_size grid
           (e.g. 16×16 = 256 tokens)
        2. Apply full self-attention on these 256 tokens (cheap)
        3. Upsample attention output back to original spatial size
        4. Add as residual to original feature map (preserves fine detail)
        5. ConvTranspose2d upsample ×2 for progressive decoding

    Memory usage:
        Full attention on 3328 tokens: 3328² × 4 bytes = 44 MB/sample
        Pooled attention on 256 tokens: 256² × 4 bytes = 0.26 MB/sample
        → 170× memory reduction, global context preserved via pooling

    Parameters
    ----------
    in_channels  : channels of input feature map
    out_channels : channels after upsample ×2
    embed_dim    : ViT token dimension (after channel projection)
    num_heads    : attention heads
    num_layers   : transformer blocks
    pool_size    : spatial size to pool to before attention (e.g. 16)
    """
    def __init__(self, in_channels, out_channels, embed_dim,
                 num_heads=4, num_layers=4, pool_size=16, dropout=0.1):
        super().__init__()
        self.pool_size = pool_size

        # Project input channels → embed_dim
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Positional embedding for pooled grid (pool_size² tokens)
        num_tokens = pool_size * pool_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks on pooled tokens
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project back to in_channels for residual addition
        self.out_proj = nn.Conv2d(embed_dim, in_channels, 1)

        # Final channel change + ×2 upsample for decoder progression
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        """x: [B, in_channels, H, W]"""
        B, C, H, W = x.shape
        P = self.pool_size

        # 1. Project channels
        feat = self.in_proj(x)                              # [B, embed_dim, H, W]
        D    = feat.shape[1]

        # 2. Pool to small fixed grid
        pooled = F.adaptive_avg_pool2d(feat, (P, P))        # [B, embed_dim, P, P]

        # 3. Flatten to token sequence
        tokens = pooled.flatten(2).transpose(1, 2)          # [B, P², embed_dim]
        tokens = tokens + self.pos_embed

        # 4. Self-attention on pooled tokens (cheap: P²×P² attention)
        tokens = self.norm(self.blocks(tokens))             # [B, P², embed_dim]

        # 5. Reshape back and upsample to original spatial size
        attn_map = tokens.transpose(1, 2).reshape(B, D, P, P)
        attn_map = F.interpolate(attn_map, (H, W),
                                 mode='bilinear', align_corners=True)

        # 6. Project back to in_channels and add as residual
        attn_out = self.out_proj(attn_map)                  # [B, in_channels, H, W]
        x = x + attn_out                                    # residual: keeps fine detail

        # 7. Upsample ×2 for decoder
        return self.upsample(x)                             # [B, out_channels, H*2, W*2]


# ══════════════════════════════════════════════════════════
# DEPTH HEAD
# ══════════════════════════════════════════════════════════

class DepthHead(nn.Module):
    def __init__(self, in_channels, max_depth=80.0):
        super().__init__()
        self.max_depth = max_depth
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(in_channels // 2, 32, 3, padding=1),          nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x, original_size):
        x = self.head(x)
        x = F.interpolate(x, original_size, mode='bilinear', align_corners=True)
        return torch.sigmoid(x) * self.max_depth


# ══════════════════════════════════════════════════════════
# FULL 3-STAGE DEPTH PREDICTION TRANSFORMER
# ══════════════════════════════════════════════════════════

class DepthPredictionTransformer(nn.Module):
    """
    3-Stage Multi-ViT Depth Prediction Transformer.

    ViT applied at three scales:
      Stage 1 (EncoderViT, 12 layers)
        → Full patch-level attention on image tokens
        → Captures global scene layout

      Stage 2 (PooledAttentionViT, 4 layers, pool=16×16)
        → Pooled attention on encoder output (H/4 features)
        → Captures depth relationships between regions
        → Memory: 256 tokens instead of 3328

      Stage 3 (PooledAttentionViT, 2 layers, pool=8×8)
        → Pooled attention on mid-resolution features (H/2)
        → Refines depth boundary transitions
        → Memory: 64 tokens instead of 13312

    Config presets:
        'tiny'  : embed=192, heads=3,  fusion=96
        'small' : embed=384, heads=6,  fusion=128  ← default
        'base'  : embed=768, heads=12, fusion=192

    GPU memory at batch=2, input=128×416:
        'tiny'  : ~2.5 GB
        'small' : ~3.5 GB   ← fits 6GB GPU
        'base'  : ~5.5 GB   ← tight on 6GB, use batch=1

    Usage:
        model = DepthPredictionTransformer(config='small', max_depth=80)
        depth = model(rgb)   # [B, 1, H, W]
    """

    CONFIGS = {
        'tiny':  dict(embed_dim=192, num_heads=3,  num_layers=12,
                      fusion_ch=96,  mid_embed=64,  dec_embed=32),
        'small': dict(embed_dim=384, num_heads=6,  num_layers=12,
                      fusion_ch=128, mid_embed=128, dec_embed=64),
        'base':  dict(embed_dim=768, num_heads=12, num_layers=12,
                      fusion_ch=192, mid_embed=192, dec_embed=96),
    }

    def __init__(self, config='small', patch_size=16,
                 max_depth=80.0, dropout=0.1):
        super().__init__()
        cfg = self.CONFIGS[config]
        D   = cfg['embed_dim']
        FC  = cfg['fusion_ch']
        ME  = cfg['mid_embed']
        DE  = cfg['dec_embed']

        # ── Stage 1: Encoder ViT ──────────────────────────────
        # Full attention on image patches (N = H/16 × W/16 = 52 tokens
        # for 128×416 — already small, no pooling needed here)
        self.encoder = EncoderViT(
            patch_size  = patch_size,
            embed_dim   = D,
            num_heads   = cfg['num_heads'],
            num_layers  = cfg['num_layers'],
            dropout     = dropout,
        )

        # Reassemble 4 tap points → 4 spatial scales
        self.reassemble = nn.ModuleList([
            Reassemble(D, FC, scale=s) for s in [4, 2, 1, 1]
        ])

        # FPN fusion → single feature map at ~H/4 resolution
        self.fusion = nn.ModuleList([FusionBlock(FC) for _ in range(4)])

        # ── Stage 2: Mid ViT (pooled, memory-safe) ────────────
        # Input:  [B, FC, H/4, W/4]  e.g. [B,128,32,104]
        # Pool to 16×16 = 256 tokens before attention
        # Output: [B, ME, H/2, W/2]
        self.mid_vit = PooledAttentionViT(
            in_channels  = FC,
            out_channels = ME,
            embed_dim    = ME,
            num_heads    = 4,
            num_layers   = 4,
            pool_size    = 16,   # 256 tokens — well within 6GB
            dropout      = dropout,
        )

        # ── Stage 3: Decoder ViT (pooled, even lighter) ───────
        # Input:  [B, ME, H/2, W/2]  e.g. [B,128,64,208]
        # Pool to 8×8 = 64 tokens before attention
        # Output: [B, DE, H, W]
        self.decoder_vit = PooledAttentionViT(
            in_channels  = ME,
            out_channels = DE,
            embed_dim    = DE,
            num_heads    = 4,
            num_layers   = 2,
            pool_size    = 8,    # 64 tokens — trivially cheap
            dropout      = dropout,
        )

        # ── Depth Head ────────────────────────────────────────
        self.head = DepthHead(DE, max_depth)

        self._print_info(config)

    def _print_info(self, config):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        mid_p = sum(p.numel() for p in self.mid_vit.parameters())
        dec_p = sum(p.numel() for p in self.decoder_vit.parameters())
        print(f"\n[DepthPredictionTransformer-{config}]")
        print(f"  Total      : {total:,} params ({train:,} trainable)")
        print(f"  Stage 1 ViT: {enc_p:,} params  (encoder, 12 layers, patch tokens)")
        print(f"  Stage 2 ViT: {mid_p:,} params  (mid,      4 layers, 16×16 pooled)")
        print(f"  Stage 3 ViT: {dec_p:,} params  (decoder,  2 layers,  8×8 pooled)")

    def forward(self, x):
        original_size = x.shape[2:]

        # ── Stage 1: Encoder ViT ──────────────────────────────
        intermediates, Hp, Wp = self.encoder(x)
        hooks = self.encoder.hooks

        fmaps = [self.reassemble[i](intermediates[hooks[i]], Hp, Wp)
                 for i in range(4)]

        # FPN fusion bottom-up: coarsest → finest
        out = self.fusion[3](fmaps[3])
        out = self.fusion[2](fmaps[2], out)
        out = self.fusion[1](fmaps[1], out)
        out = self.fusion[0](fmaps[0], out)
        # out: [B, FC, ~H/4, ~W/4]

        # ── Stage 2: Mid ViT (pooled 16×16) ──────────────────
        # Attends globally over 256 pooled tokens → depth reasoning
        out = self.mid_vit(out)
        # out: [B, ME, ~H/2, ~W/2]

        # ── Stage 3: Decoder ViT (pooled 8×8) ────────────────
        # Attends globally over 64 pooled tokens → boundary refinement
        out = self.decoder_vit(out)
        # out: [B, DE, ~H, ~W]

        # ── Depth Head ────────────────────────────────────────
        return self.head(out, original_size)


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    model = DepthPredictionTransformer(config='small', max_depth=80)
    print("=" * 60)

    x   = torch.randn(2, 3, 128, 416)
    out = model(x)
    print(f"\nInput : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Range : [{out.min():.3f}, {out.max():.3f}]")

    # Memory estimate
    params_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024**2
    print(f"Model weights: {params_mb:.1f} MB")