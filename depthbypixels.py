import matplotlib.pyplot as plt
import cv2
import numpy as np

# ── Load depth map ────────────────────────────────────────
# Use KITTI raw depth PNG (uint16), NOT a colorized jpg
depth_raw = cv2.imread("depth_map.png", cv2.IMREAD_UNCHANGED)

if depth_raw is None:
    raise FileNotFoundError("depth_map.png not found")

# If accidentally loaded a colorized RGB image, convert to grayscale
if depth_raw.ndim == 3:
    print("WARNING: Image has 3 channels — converting to grayscale.")
    print("For real depth values, use the original KITTI uint16 .png file.")
    depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

# KITTI format: pixel value / 256.0 = metres
depth_m = depth_raw.astype(np.float32) / 256.0

print(f"Depth map shape : {depth_m.shape}")
print(f"Depth range     : {depth_m[depth_m > 0].min():.2f} m  →  {depth_m.max():.2f} m")
print(f"Valid pixels    : {(depth_m > 0).mean() * 100:.1f}%")

# ── Interactive click ─────────────────────────────────────
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return                          # clicked outside image
    x = int(event.xdata)
    y = int(event.ydata)

    # Guard against out-of-bounds
    h, w = depth_m.shape
    if not (0 <= x < w and 0 <= y < h):
        return

    d = float(depth_m[y, x])           # scalar float — fixes the TypeError

    if d == 0:
        print(f"Pixel ({x}, {y}) → No depth data (invalid/sparse)")
    else:
        print(f"Pixel ({x}, {y}) → Depth: {d:.2f} m")

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(depth_m, cmap='plasma')
plt.colorbar(im, ax=ax, label='Depth (m)')
ax.set_title("Click anywhere to read depth value")
fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()