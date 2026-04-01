"""
vit_model.py
─────────────────────────────────────────────────────────────
Pure Vision Transformer (ViT) Depth Estimation Model

Architecture: Patch Embedding → ViT Encoder → Progressive Decoder
─────────────────────────────────────────────────────────────

OVERVIEW
────────
Unlike CNN which slides a small kernel across the image,
ViT treats the image as a sequence of non-overlapping patches
and applies full self-attention across ALL patches at once.

PIPELINE
────────
  RGB image [B, 3, H, W]
       │
  PatchEmbedding
    • split into P×P patches (patch_size=16)
    • flatten each patch → linear project → embed_dim
    → [B, num_patches, embed_dim]   num_patches = (H/16)×(W/16)
       │
  + CLS token  [B, 1+num_patches, embed_dim]
  + Positional embedding (learnable)
       │
  N × TransformerBlock
    • LayerNorm → Multi-Head Self-Attention → residual
    • LayerNorm → FFN (MLP) → residual
    → [B, 1+num_patches, embed_dim]
       │
  ViT Decoder
    • remove CLS, reshape patches back to 2D grid
    • progressive upsampling with ConvTranspose2d + Conv
    → [B, 1, H, W]   depth map [0..max_depth]

WHY ViT FOR DEPTH?
──────────────────
• Self-attention is O(N²) over all patches simultaneously
  → every patch can attend to every other patch in ONE layer
• Great for global context: sky is always far, objects near
  the bottom of a road image are closer
• Weakness vs CNN: needs more data, less translation-equivariant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and linearly embed each.

    patch_size=16 on a 128×416 image → 8×26 = 208 patches
    Each patch: 16×16×3 = 768 values → projected to embed_dim

    Uses Conv2d(kernel=patch_size, stride=patch_size) which is
    mathematically equivalent to splitting then doing linear but faster.
    """
    def __init__(self, in_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Pad so H and W are divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)          # [B, embed_dim, H/P, W/P]
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # [B, N, embed_dim]
        return self.norm(x), Hp, Wp


class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

        Attention(Q,K,V) = softmax( QKᵀ/√d ) V

    All heads computed in parallel via a single fused QKV projection.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each [B,H,N,d]

        attn = (q @ k.transpose(-2, -1)) * self.scale      # [B,H,N,N]
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """
    Standard ViT encoder block (Pre-LN):
        x ← x + MHSA(LN(x))
        x ← x + FFN(LN(x))
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════
# ViT ENCODER
# ══════════════════════════════════════════════════════════

class ViTEncoder(nn.Module):
    """
    Full ViT image encoder.

    Adds a CLS token (classification token) that aggregates global
    image information through attention with all patch tokens.
    Positional embeddings are learnable.

    Returns:
        patch_tokens : [B, N, embed_dim]   (N = H/P × W/P patches)
        Hp, Wp       : patch grid dimensions
    """
    def __init__(self,
                 in_channels: int  = 3,
                 patch_size:  int  = 16,
                 embed_dim:   int  = 768,
                 num_heads:   int  = 12,
                 num_layers:  int  = 12,
                 mlp_ratio:   float = 4.0,
                 dropout:     float = 0.1,
                 max_patches: int  = 512):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        # CLS token — a learnable vector prepended to the patch sequence
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding: 1 for CLS + max_patches for patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, 1 + max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop    = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """Interpolate positional embedding if patch count differs from stored."""
        pos_cls  = self.pos_embed[:, :1, :]          # CLS pos
        pos_patch = self.pos_embed[:, 1:, :]         # patch pos [1, stored_N, D]
        stored_N  = pos_patch.shape[1]
        if N == stored_N:
            return self.pos_embed
        # 1-D interpolation
        pos_patch = pos_patch.transpose(1, 2)        # [1, D, stored_N]
        pos_patch = F.interpolate(pos_patch, size=N, mode='linear', align_corners=False)
        pos_patch = pos_patch.transpose(1, 2)        # [1, N, D]
        return torch.cat([pos_cls, pos_patch], dim=1)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        tokens, Hp, Wp = self.patch_embed(x)    # [B, N, D]
        N = tokens.shape[1]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # [B, 1+N, D]

        # Add positional embedding
        tokens = tokens + self._interpolate_pos_embed(tokens, N)
        tokens = self.pos_drop(tokens)

        # Transformer blocks
        tokens = self.norm(self.blocks(tokens))    # [B, 1+N, D]

        # Return patch tokens only (drop CLS)
        return tokens[:, 1:, :], Hp, Wp            # [B, N, D]


# ══════════════════════════════════════════════════════════
# ViT DECODER (progressive upsampling)
# ══════════════════════════════════════════════════════════

class ViTDepthDecoder(nn.Module):
    """
    Reshape patch tokens back to 2D, then progressively upsample to
    full resolution using transposed convolutions.

    The patch_size determines how many upsampling steps are needed:
        patch_size=16 → 4 steps of ×2 = ×16 upsampling total
        patch_size=8  → 3 steps
    """
    def __init__(self, embed_dim: int = 768,
                 patch_size: int = 16,
                 max_depth: float = 80.0):
        super().__init__()
        self.patch_size = patch_size
        self.max_depth  = max_depth

        # Number of ×2 upsampling steps = log2(patch_size)
        num_ups = int(math.log2(patch_size))  # 4 for patch_size=16

        layers = []
        in_ch  = embed_dim
        out_ch = embed_dim // 2
        for i in range(num_ups):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ELU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(inplace=True),
            ]
            in_ch  = out_ch
            out_ch = max(out_ch // 2, 16)

        self.upsample = nn.Sequential(*layers)
        self.out      = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, tokens: torch.Tensor,
                Hp: int, Wp: int,
                original_size) -> torch.Tensor:
        """
        tokens : [B, Hp*Wp, embed_dim]
        Hp, Wp : patch grid dimensions
        """
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)  # [B, D, Hp, Wp]
        x = self.upsample(x)
        x = F.interpolate(x, original_size, mode='bilinear', align_corners=True)
        return torch.sigmoid(self.out(x)) * self.max_depth


# ══════════════════════════════════════════════════════════
# FULL ViT DEPTH MODEL
# ══════════════════════════════════════════════════════════

class ViTDepthModel(nn.Module):
    """
    Pure Vision Transformer depth estimation model.

    Config presets:
        'tiny'  : embed=192, heads=3,  layers=12
        'small' : embed=384, heads=6,  layers=12
        'base'  : embed=768, heads=12, layers=12  ← default

    Usage:
        model = ViTDepthModel(config='small', max_depth=80)
        depth = model(rgb_tensor)   # [B, 1, H, W]
    """

    CONFIGS = {
        'tiny':  dict(embed_dim=192, num_heads=3,  num_layers=12),
        'small': dict(embed_dim=384, num_heads=6,  num_layers=12),
        'base':  dict(embed_dim=768, num_heads=12, num_layers=12),
    }

    def __init__(self,
                 config:     str   = 'small',
                 patch_size: int   = 16,
                 max_depth:  float = 80.0,
                 dropout:    float = 0.1):
        super().__init__()
        cfg = self.CONFIGS[config]

        self.encoder = ViTEncoder(
            in_channels = 3,
            patch_size  = patch_size,
            embed_dim   = cfg['embed_dim'],
            num_heads   = cfg['num_heads'],
            num_layers  = cfg['num_layers'],
            mlp_ratio   = 4.0,
            dropout     = dropout,
        )
        self.decoder = ViTDepthDecoder(
            embed_dim  = cfg['embed_dim'],
            patch_size = patch_size,
            max_depth  = max_depth,
        )
        self._print_info(config)

    def _print_info(self, config: str):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ViTDepthModel-{config}] {total:,} params | {trainable:,} trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[2:]
        tokens, Hp, Wp = self.encoder(x)
        return self.decoder(tokens, Hp, Wp, original_size)


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    model = ViTDepthModel(config='small', patch_size=16, max_depth=80)
    x     = torch.randn(2, 3, 128, 416)
    out   = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Range : [{out.min():.3f}, {out.max():.3f}]")