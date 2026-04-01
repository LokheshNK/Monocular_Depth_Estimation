"""
train_cnn.py
─────────────────────────────────────────────────────────────
Train the pure CNN depth model (VGG16 encoder + U-Net decoder).
Architecture lives in: cnn_model.py → CNNDepthModel
Saves → best_cnn_model.pth

Run:  python train_cnn.py
─────────────────────────────────────────────────────────────
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# ── Import model and sequence finder ──────────────────────
from cnn_model       import CNNDepthModel
from find_sequences  import auto_find_sequences

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Auto-discovers all sequences — just add more KITTI folders
# and they are picked up automatically. Preview: python find_sequences.py
SEQUENCE_PAIRS = auto_find_sequences(BASE_DIR, verbose=True)

IMG_SIZE   = (128, 416)   # (H, W)
BATCH_SIZE = 4
EPOCHS     = 40
MAX_DEPTH  = 80.0
LR_ENC     = 5e-6        # VGG16 pretrained encoder — very gentle
LR_DEC     = 1e-4        # U-Net decoder — normal
VAL_SPLIT  = 0.1
SAVE_PATH  = "best_cnn_model.pth"   # ← CNN checkpoint


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class KITTIDepthDataset(Dataset):
    """
    Loads matched RGB + sparse-depth pairs from KITTI.
    Supports multiple sequence pairs (auto-discovered).
    Augmentation: flip, brightness/contrast, random crop — all joint.
    """

    def __init__(self, sequence_pairs, img_size=(128, 416), augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []

        for rgb_dir, depth_dir in sequence_pairs:
            if not os.path.isdir(rgb_dir):
                print(f"  SKIP (missing): {rgb_dir}"); continue
            if not os.path.isdir(depth_dir):
                print(f"  SKIP (missing): {depth_dir}"); continue

            common = sorted(set(os.listdir(rgb_dir)) & set(os.listdir(depth_dir)))
            if not common:
                print(f"  WARNING: no matching files in {rgb_dir}"); continue

            for f in common:
                self.samples.append((
                    os.path.join(rgb_dir,   f),
                    os.path.join(depth_dir, f),
                ))
            print(f"  {len(common)} pairs  ←  {os.path.basename(rgb_dir)}")

        print(f"\nTotal samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb   = Image.open(rgb_path).convert("RGB")
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 256.0

        H, W  = self.img_size
        rgb   = rgb.resize((W, H), Image.BILINEAR)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Joint horizontal flip
            if random.random() > 0.5:
                rgb   = TF.hflip(rgb)
                depth = depth[:, ::-1].copy()
            # RGB-only colour jitter
            rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_contrast(rgb,   random.uniform(0.8, 1.2))
            # Joint random crop
            ch   = random.randint(int(H * 0.9), H)
            cw   = random.randint(int(W * 0.9), W)
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            rgb   = TF.crop(rgb, top, left, ch, cw).resize((W, H), Image.BILINEAR)
            depth = cv2.resize(depth[top:top+ch, left:left+cw],
                               (W, H), interpolation=cv2.INTER_NEAREST)

        rgb_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])(rgb)

        return rgb_t, torch.from_numpy(depth).unsqueeze(0)


class AugmentedSubset(torch.utils.data.Dataset):
    """Toggle augmentation flag on a random_split subset."""
    def __init__(self, subset, augment):
        self.subset  = subset
        self.augment = augment

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        orig = self.subset.dataset.augment
        self.subset.dataset.augment = self.augment
        item = self.subset[idx]
        self.subset.dataset.augment = orig
        return item


# ══════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════

class BerHuLoss(nn.Module):
    """Reverse Huber: L1 for small errors, L2 for large."""
    def __init__(self, threshold=0.2):
        super().__init__(); self.t = threshold

    def forward(self, pred, gt):
        mask = gt > 0
        if mask.sum() == 0:
            return torch.tensor(0., device=pred.device, requires_grad=True)
        p, g = pred[mask], gt[mask]
        d = torch.abs(p - g); c = self.t * d.max().detach()
        return torch.where(d <= c, d, (d**2 + c**2) / (2*c + 1e-8)).mean()


class SSIMLoss(nn.Module):
    """Structural similarity — reduces blurriness."""
    def __init__(self, ws=11):
        super().__init__(); self.ws = ws
        coords = torch.arange(ws).float() - ws // 2
        g = torch.exp(-(coords**2) / (2*1.5**2)); g /= g.sum()
        self.register_buffer("w", (g.unsqueeze(0)*g.unsqueeze(1)).unsqueeze(0).unsqueeze(0))

    def forward(self, pred, gt):
        m = (gt > 0).float(); pred = pred*m; gt = gt*m
        w = self.w.to(pred.device); pad = self.ws // 2
        mu1 = F.conv2d(pred, w, padding=pad); mu2 = F.conv2d(gt, w, padding=pad)
        s1  = F.conv2d(pred*pred, w, padding=pad) - mu1**2
        s2  = F.conv2d(gt*gt,     w, padding=pad) - mu2**2
        s12 = F.conv2d(pred*gt,   w, padding=pad) - mu1*mu2
        C1, C2 = 1e-4, 9e-4
        ssim = (2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2) + 1e-8)
        return (1 - ssim.mean()) / 2


class EdgeGradientLoss(nn.Module):
    """Sobel gradient matching — sharpens depth edges."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1,1,3,3))
        self.register_buffer("ky", ky.view(1,1,3,3))

    def forward(self, pred, gt):
        m = (gt > 0).float()
        gxp = F.conv2d(pred, self.kx.to(pred.device), padding=1)
        gyp = F.conv2d(pred, self.ky.to(pred.device), padding=1)
        gxg = F.conv2d(gt,   self.kx.to(gt.device),   padding=1)
        gyg = F.conv2d(gt,   self.ky.to(gt.device),   padding=1)
        return ((torch.abs(gxp-gxg) + torch.abs(gyp-gyg)) * m).mean()


class CombinedLoss(nn.Module):
    """BerHu(×1.0) + SSIM(×0.5) + Edge(×0.5)"""
    def __init__(self):
        super().__init__()
        self.berhu = BerHuLoss(); self.ssim = SSIMLoss(); self.edge = EdgeGradientLoss()

    def forward(self, pred, gt):
        lb = self.berhu(pred, gt)
        ls = self.ssim(pred, gt)
        le = self.edge(pred, gt)
        return lb + 0.5*ls + 0.5*le, lb.item(), ls.item(), le.item()


# ══════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════

print("\nLoading dataset…")
full_ds = KITTIDepthDataset(SEQUENCE_PAIRS, IMG_SIZE)
if len(full_ds) == 0:
    raise RuntimeError("No samples found. Run: python find_sequences.py")

val_n = max(1, int(len(full_ds) * VAL_SPLIT))
tr_n  = len(full_ds) - val_n
tr_ds, val_ds = random_split(full_ds, [tr_n, val_n])

tr_loader  = DataLoader(AugmentedSubset(tr_ds,  augment=True),
                        BATCH_SIZE, shuffle=True,  num_workers=0,
                        drop_last=True,  pin_memory=True)
val_loader = DataLoader(AugmentedSubset(val_ds, augment=False),
                        BATCH_SIZE, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)

print(f"Train: {tr_n}  |  Val: {val_n}  |  Batches/epoch: {len(tr_loader)}")


# ══════════════════════════════════════════════════════════
# MODEL — instantiated from cnn_model.py
# ══════════════════════════════════════════════════════════

model = CNNDepthModel(max_depth=MAX_DEPTH, pretrained=True).to(device)
criterion = CombinedLoss()

# Separate LRs: pretrained VGG encoder gets much smaller LR
optimizer = torch.optim.AdamW([
    {'params': model.encoder.block3.parameters(), 'lr': LR_ENC},
    {'params': model.encoder.block4.parameters(), 'lr': LR_ENC},
    {'params': model.encoder.block5.parameters(), 'lr': LR_ENC},
    {'params': model.decoder.parameters(),        'lr': LR_DEC},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════

best_val = float('inf')
hist = {'train': [], 'val': [], 'berhu': [], 'ssim': [], 'edge': []}

print(f"\n{'='*60}")
print(f"  CNN MODEL  |  {EPOCHS} epochs  |  {device}")
print(f"  Loss = BerHu + SSIM×0.5 + Edge×0.5")
print(f"  Checkpoint → {SAVE_PATH}")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):

    # ── Train ────────────────────────────────────────────
    model.train()
    tl = tb = ts = te = n = 0

    for rgb, depth in tr_loader:
        rgb, depth = rgb.to(device), depth.to(device)
        loss, lb, ls, le = criterion(model(rgb), depth)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tl += loss.item(); tb += lb; ts += ls; te += le; n += 1

    scheduler.step()
    at = tl / max(n, 1)

    # ── Validate ─────────────────────────────────────────
    model.eval()
    vl = nv = 0
    with torch.no_grad():
        for rgb, depth in val_loader:
            loss, *_ = criterion(model(rgb.to(device)), depth.to(device))
            vl += loss.item(); nv += 1
    av = vl / max(nv, 1)

    hist['train'].append(at);           hist['val'].append(av)
    hist['berhu'].append(tb/max(n,1));  hist['ssim'].append(ts/max(n,1))
    hist['edge'].append(te/max(n,1))

    tag = ""
    if av < best_val:
        best_val = av
        torch.save(model.state_dict(), SAVE_PATH)
        tag = "  ← saved"

    print(f"Ep {epoch+1:3d}/{EPOCHS}  |  "
          f"Train {at:.4f}  Val {av:.4f}  |  "
          f"BerHu {tb/max(n,1):.3f}  SSIM {ts/max(n,1):.3f}  Edge {te/max(n,1):.3f}"
          f"{tag}")

print(f"\nBest val loss: {best_val:.4f}  →  {SAVE_PATH}")


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(hist['train'], label='Train'); axes[0].plot(hist['val'], label='Val')
axes[0].set_title('CNN — Total Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(hist['berhu'], label='BerHu')
axes[1].plot(hist['ssim'],  label='SSIM')
axes[1].plot(hist['edge'],  label='Edge')
axes[1].set_title('Loss Components'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('cnn_loss.png', dpi=150); plt.show()
print("Saved → cnn_loss.png")


# ══════════════════════════════════════════════════════════
# EVALUATION + VISUALISATION
# ══════════════════════════════════════════════════════════

model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()
abs_rel, rmse_list, a1_list = [], [], []
mean_t = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std_t  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

with torch.no_grad():
    for rgb, depth in DataLoader(val_ds, batch_size=1):
        rgb, depth = rgb.to(device), depth.to(device)
        pred = model(rgb)
        mask = depth > 0
        if mask.sum() == 0: continue
        p = pred[mask].clamp(1e-3, MAX_DEPTH)
        g = depth[mask].clamp(1e-3, MAX_DEPTH)
        abs_rel.append(((torch.abs(p-g)/g).mean()).item())
        rmse_list.append(torch.sqrt(((p-g)**2).mean()).item())
        a1_list.append((torch.max(p/g, g/p) < 1.25).float().mean().item())

print(f"\n{'='*60}")
print(f"  CNN — Final Validation Metrics")
print(f"{'='*60}")
print(f"  Abs Rel Error : {np.mean(abs_rel):.4f}  (↓ better)")
print(f"  RMSE          : {np.mean(rmse_list):.4f}  (↓ better)")
print(f"  δ < 1.25      : {np.mean(a1_list):.4f}  (↑ better)")

# Visualise 4 validation samples
num_vis = min(4, len(val_ds))
fig, axes = plt.subplots(num_vis, 4, figsize=(20, 5*num_vis))
if num_vis == 1: axes = axes[np.newaxis, :]

with torch.no_grad():
    for i in range(num_vis):
        rgb_t, depth_t = val_ds[i]
        pred_t = model(rgb_t.unsqueeze(0).to(device))

        pred_np = pred_t.squeeze().cpu().numpy()
        gt_np   = depth_t.squeeze().numpy()
        rgb_disp= (rgb_t * std_t + mean_t).clamp(0,1).permute(1,2,0).numpy()
        error   = np.abs(pred_np - gt_np) * (gt_np > 0).astype(float)

        axes[i,0].imshow(rgb_disp); axes[i,0].set_title('RGB'); axes[i,0].axis('off')
        im1=axes[i,1].imshow(gt_np,   cmap='plasma',vmin=0,vmax=80); axes[i,1].set_title('Ground Truth');  axes[i,1].axis('off'); plt.colorbar(im1,ax=axes[i,1],fraction=0.046)
        im2=axes[i,2].imshow(pred_np, cmap='plasma',vmin=0,vmax=80); axes[i,2].set_title('Predicted');     axes[i,2].axis('off'); plt.colorbar(im2,ax=axes[i,2],fraction=0.046)
        im3=axes[i,3].imshow(error,   cmap='hot',   vmin=0,vmax=15); axes[i,3].set_title('Abs Error');     axes[i,3].axis('off'); plt.colorbar(im3,ax=axes[i,3],fraction=0.046)

plt.tight_layout(); plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved → cnn_predictions.png")