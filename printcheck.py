import cv2
import numpy as np
import matplotlib.pyplot as plt

depth_img = cv2.imread(
    r"D:/SEM-6/depth estimation/depth dataset/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000102.png",
    cv2.IMREAD_UNCHANGED
)

depth = depth_img.astype(np.float32) / 256.0

# remove invalid pixels
depth[depth == 0] = np.nan

plt.figure(figsize=(10,4))
plt.imshow(depth, cmap='plasma')
plt.colorbar(label="Depth (meters)")
plt.title("KITTI Ground Truth Depth")
plt.show()

import os

rgb_path = r"D:\SEM-6\depth estimation\depth dataset\raw_rgb\2011_09_26\2011_09_26_drive_0001_sync\image_02\data"
depth_path = r"D:\SEM-6\depth estimation\depth dataset\data_depth_annotated\train\2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02"

rgb_files = set(os.listdir(rgb_path))
depth_files = set(os.listdir(depth_path))

print("RGB images:", len(rgb_files))
print("Depth maps:", len(depth_files))
print("Matching:", len(rgb_files & depth_files))
