"""
hybrid_model.py
─────────────────────────────────────────────────────────────
Hybrid CNN + ViT Depth Estimation Model

Composes CNNEncoder and ViTBottleneck via object composition.
Imports directly from cnn_model.py and vit_model.py.
─────────────────────────────────────────────────────────────

ARCHITECTURE
────────────

  RGB [B, 3, H, W]
       │
  ┌────▼──────────────────────────────────────────┐
  │  CNNEncoder (from cnn_model.py)                │
  │  enc1 [B,  64, H/2,  W/2 ]  ← edges           │
  │  enc2 [B, 128, H/4,  W/4 ]  ← textures        │
  │  enc3 [B, 256, H/8,  W/8 ]  ← parts           │
  │  enc4 [B, 512, H/16, W/16]  ← objects         │
  │  enc5 [B, 512, H/32, W/32]  ← scene layout    │
  └──────────────────────┬────────────────────────┘
                         │ enc5
  ┌──────────────────────▼────────────────────────┐
  │  ViTBottleneck (from vit_model.py components)  │
  │  • 1×1 conv: 512 → embed_dim                  │
  │  • flatten → N tokens (H/32 × W/32)           │
  │  • learnable positional embeddings             │
  │  • num_layers × TransformerBlock (MHSA + FFN) │
  │  • LayerNorm                                   │
  │  • 1×1 conv: embed_dim → 512                  │
  └──────────────────────┬────────────────────────┘
                         │ globally-enriched enc5
  ┌──────────────────────▼────────────────────────┐
  │  CNNDecoder with skip connections              │
  │  up5 + skip(enc4) → [B, 128, H/16, W/16]     │
  │  up4 + skip(enc3) → [B,  64, H/8,  W/8 ]     │
  │  up3 + skip(enc2) → [B,  32, H/4,  W/4 ]     │
  │  up2 + skip(enc1) → [B,  16, H/2,  W/2 ]     │
  │  up1              → [B,   1, H,    W   ]      │
  └──────────────────────────────────────────────┘
       │
  sigmoid × max_depth → depth map [0..max_depth]

WHY HYBRID?
───────────
  CNN  = strong LOCAL features (pretrained VGG16, data-efficient)
  ViT  = GLOBAL context (full receptive field at bottleneck)
  Hybrid = best of both worlds, inserted at smallest feature map
           (H/32 × W/32) where global attention is cheapest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import building blocks from sibling modules ───────────
from cnn_model import CNNEncoder, CNNDecoder
from vit_model import MultiHeadSelfAttention, TransformerBlock


# ══════════════════════════════════════════════════════════
# ViT BOTTLENECK  (re-used from vit_model components)
# ══════════════════════════════════════════════════════════

class ViTBottleneck(nn.Module):
    """
    A compact ViT module designed to sit between a CNN encoder and decoder.

    It applies global self-attention across the (small) spatial grid of
    the deepest CNN feature map, then restores the same channel/spatial
    dimensions so the existing CNN decoder can consume it unchanged.

    For a 128×416 input with VGG16:
        enc5 spatial size = 4×13 = 52 tokens
        attention matrix  = 52×52 (trivially cheap)

    Parameters
    ----------
    in_channels   : channels from CNN encoder  (512 for VGG16)
    out_channels  : channels for CNN decoder   (512 to match enc5)
    spatial_size  : (H, W) of enc5              e.g. (4, 13)
    embed_dim     : ViT internal token width    (256 default)
    num_heads     : parallel attention heads    (embed_dim divisible)
    num_layers    : transformer depth
    """
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 spatial_size: tuple,
                 embed_dim:    int   = 256,
                 num_heads:    int   = 8,
                 num_layers:   int   = 4,
                 mlp_ratio:    float = 4.0,
                 dropout:      float = 0.1):
        super().__init__()
        H, W = spatial_size
        self.H, self.W = H, W
        num_tokens = H * W

        # Channel reduction: CNN channels → ViT embed_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Learnable positional embedding — one per spatial position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stack of transformer blocks (reused from vit_model.py)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Channel restoration: embed_dim → decoder channels
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"  [ViTBottleneck] {n:,} params | "
              f"{H}×{W}={num_tokens} tokens | embed={embed_dim} | "
              f"heads={num_heads} | layers={num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Reduce channels
        x = self.input_proj(x)                        # [B, embed_dim, H, W]

        # 2. Flatten to token sequence
        x = x.flatten(2).transpose(1, 2)              # [B, H*W, embed_dim]

        # 3. Add positional embeddings (interpolate if spatial size changed)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos = self.pos_embed.reshape(1, self.H, self.W, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, (H, W), mode='bilinear', align_corners=True)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed
        x = x + pos                                    # [B, H*W, embed_dim]

        # 4. Global self-attention
        x = self.norm(self.blocks(x))                  # [B, H*W, embed_dim]

        # 5. Reshape back to spatial map
        x = x.transpose(1, 2).reshape(B, -1, H, W)    # [B, embed_dim, H, W]

        # 6. Restore channels
        return self.output_proj(x)                     # [B, out_channels, H, W]


# ══════════════════════════════════════════════════════════
# HYBRID MODEL — object composition
# ══════════════════════════════════════════════════════════

class HybridDepthModel(nn.Module):
    """
    Hybrid CNN + ViT depth estimation model.

    Composed from:
        CNNEncoder    (cnn_model.py)
        ViTBottleneck (vit_model.py components)
        CNNDecoder    (cnn_model.py)

    Usage:
        model = HybridDepthModel(max_depth=80)
        depth = model(rgb_tensor)   # [B, 1, H, W]

    To change ViT capacity, adjust vit_embed_dim / vit_num_layers:
        HybridDepthModel(vit_embed_dim=512, vit_num_layers=6)
    """
    def __init__(self,
                 max_depth:      float = 80.0,
                 pretrained_cnn: bool  = True,
                 vit_embed_dim:  int   = 256,
                 vit_num_heads:  int   = 8,
                 vit_num_layers: int   = 4,
                 vit_spatial:    tuple = (4, 13),  # enc5 size for 128×416 input
                 dropout:        float = 0.1):
        super().__init__()

        print("[HybridDepthModel] Building components…")

        # ── 1. CNN Encoder (from cnn_model.py) ────────────────
        self.cnn_encoder = CNNEncoder(pretrained=pretrained_cnn)

        # ── 2. ViT Bottleneck (uses vit_model.py TransformerBlock) ─
        self.vit_bottleneck = ViTBottleneck(
            in_channels  = 512,
            out_channels = 512,
            spatial_size = vit_spatial,
            embed_dim    = vit_embed_dim,
            num_heads    = vit_num_heads,
            num_layers   = vit_num_layers,
            dropout      = dropout,
        )

        # ── 3. CNN Decoder (from cnn_model.py) ────────────────
        self.cnn_decoder = CNNDecoder(max_depth=max_depth)

        self._print_info()

    def _print_info(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[HybridDepthModel] Total: {total:,} | Trainable: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — CNN extracts local multi-scale features
        e1, e2, e3, e4, e5 = self.cnn_encoder(x)

        # Step 2 — ViT adds global context to deepest feature map
        #          enc5 is smallest (4×13 for 128×416), making attention cheap
        e5_global = self.vit_bottleneck(e5)   # [B, 512, 4, 13]

        # Step 3 — CNN Decoder reconstructs depth using skip connections
        #          Note: we pass e5_global in place of original e5
        return self.cnn_decoder(e1, e2, e3, e4, e5_global, x.shape[2:])


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    model = HybridDepthModel(max_depth=80, pretrained_cnn=False)
    print("=" * 55)

    x   = torch.randn(2, 3, 128, 416)
    out = model(x)
    print(f"\nInput : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Range : [{out.min():.3f}, {out.max():.3f}]")