"""
cnn_model.py
─────────────────────────────────────────────────────────────
Pure CNN Depth Estimation Model

Architecture: VGG16 Encoder  +  U-Net style Decoder
─────────────────────────────────────────────────────────────

ENCODER (VGG16 pretrained on ImageNet):
  Input  : [B, 3,   H,   W]
  enc1   : [B,  64, H/2, W/2]   ← edges, colours
  enc2   : [B, 128, H/4, W/4]   ← textures
  enc3   : [B, 256, H/8, W/8]   ← parts
  enc4   : [B, 512,H/16,W/16]   ← objects
  enc5   : [B, 512,H/32,W/32]   ← scene layout

DECODER (skip connections from encoder):
  up5(enc5)          : [B, 256, H/16, W/16]
  up4(cat+enc4)      : [B, 128, H/8,  W/8 ]
  up3(cat+enc3)      : [B,  64, H/4,  W/4 ]
  up2(cat+enc2)      : [B,  32, H/2,  W/2 ]
  up1(cat+enc1)      : [B,  16, H,    W   ]
  out                : [B,   1, H,    W   ]   ← depth map [0..max_depth]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    VGG16 feature pyramid split into 5 resolution levels.
    Each level halves spatial dimensions and increases channels.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained).features

        self.block1 = vgg[:5]    # 3   →  64 ch, stride /2
        self.block2 = vgg[5:10]  # 64  → 128 ch, stride /4
        self.block3 = vgg[10:17] # 128 → 256 ch, stride /8
        self.block4 = vgg[17:24] # 256 → 512 ch, stride /16
        self.block5 = vgg[24:31] # 512 → 512 ch, stride /32

        # Freeze first two blocks — they already know edges/colours
        for p in self.block1.parameters(): p.requires_grad = False
        for p in self.block2.parameters(): p.requires_grad = False

    def forward(self, x):
        e1 = self.block1(x)
        e2 = self.block2(e1)
        e3 = self.block3(e2)
        e4 = self.block4(e3)
        e5 = self.block5(e4)
        return e1, e2, e3, e4, e5   # pyramid of features


class CNNDecoder(nn.Module):
    """
    U-Net style decoder.
    Each level: upsample ×2 → concatenate skip → refine with convs.
    """
    def __init__(self, max_depth: float = 80.0):
        super().__init__()
        self.max_depth = max_depth

        self.up5 = self._upblock(512,       256)   # enc5 only
        self.up4 = self._upblock(256 + 512, 128)   # + enc4 skip
        self.up3 = self._upblock(128 + 256,  64)   # + enc3 skip
        self.up2 = self._upblock( 64 + 128,  32)   # + enc2 skip
        self.up1 = self._upblock( 32 +  64,  16)   # + enc1 skip
        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    @staticmethod
    def _upblock(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, ref.shape[2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, e1, e2, e3, e4, e5, original_size):
        d5 = self._match(self.up5(e5), e4)
        d4 = self._match(self.up4(torch.cat([d5, e4], dim=1)), e3)
        d3 = self._match(self.up3(torch.cat([d4, e3], dim=1)), e2)
        d2 = self._match(self.up2(torch.cat([d3, e2], dim=1)), e1)
        d1 = self.up1(torch.cat([d2, e1], dim=1))
        d1 = F.interpolate(d1, original_size, mode='bilinear', align_corners=True)
        return torch.sigmoid(self.out(d1)) * self.max_depth


class CNNDepthModel(nn.Module):
    """
    Full CNN depth estimation model.

    Usage:
        model = CNNDepthModel(max_depth=80, pretrained=True)
        depth = model(rgb_tensor)   # [B, 1, H, W]
    """
    def __init__(self, max_depth: float = 80.0, pretrained: bool = True):
        super().__init__()
        self.encoder = CNNEncoder(pretrained=pretrained)
        self.decoder = CNNDecoder(max_depth=max_depth)
        self._print_info()

    def _print_info(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CNNDepthModel] {total:,} params | {trainable:,} trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4, e5 = self.encoder(x)
        return self.decoder(e1, e2, e3, e4, e5, x.shape[2:])


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    model = CNNDepthModel(max_depth=80, pretrained=False)
    x     = torch.randn(2, 3, 128, 416)
    out   = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")   # [2, 1, 128, 416]
    print(f"Range : [{out.min():.3f}, {out.max():.3f}]")