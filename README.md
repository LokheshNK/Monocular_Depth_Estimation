# Monocular Depth Estimation

## Overview

`Monocular Depth Estimation` predicts per-pixel depth from single RGB images.
This repository includes models, training scripts, and evaluation tools for depth estimation on KITTI-style datasets.

## Project Structure

- `app.py`: inference/demo script (image -> depth prediction)
- `cnn_model.py`: CNN-based depth estimator
- `vit_model.py`: Vision Transformer depth estimator
- `hybrid_model.py`: CNN + Transformer hybrid model
- `depth_prediction_transformer.py`: DPT-inspired depth transformer model
- `train_2cnn.py`: train CNN model
- `train_dpt.py`: train DPT transformer model
- `train_hybrid.py`: train hybrid model
- `depthbypixels.py`: depth map utilities / post-processing
- `find_sequences.py`: KITTI sequence utility loader
- `printcheck.py`: debug / metric print helpers
- `best_cnn_model.pth`, `best_dpt_model.pth`, `best_hybrid_model.pth`: checkpoint weights

Data folders:
- `data_depth_annotated/train/`: training sequences
- `data_depth_annotated/val/`: validation sequences
- `raw_rgb/`: raw RGB frames

## Requirements

1. Python 3.8+
2. pip packages:
   - torch
   - torchvision
   - numpy
   - opencv-python
   - tqdm
   - matplotlib
   - scipy
   - Pillow

Install with:

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not present, create it with the list above.

## Quick Start

### 1. Prepare dataset

- Use KITTI-style directory layout with aligned RGB and depth frames.
- Put folders into `data_depth_annotated/train` and `data_depth_annotated/val`.

### 2. Train CNN model

```bash
python train_2cnn.py --data_dir data_depth_annotated --epochs 30 --batch_size 8 --lr 1e-4
```

### 3. Train DPT transformer model

```bash
python train_dpt.py --data_dir data_depth_annotated --epochs 20 --batch_size 4 --lr 2e-5
```

### 4. Train hybrid model

```bash
python train_hybrid.py --data_dir data_depth_annotated --epochs 25 --batch_size 6 --lr 1e-4
```

### 5. Inference

```bash
python app.py --input_image raw_rgb/2011_09_26/0000000000.png --checkpoint best_cnn_model.pth --output output_depth.png
```

## Checkpoints

- `best_cnn_model.pth`
- `best_dpt_model.pth`
- `best_hybrid_model.pth`

## Evaluation Metrics

- RMSE
- Abs Rel
- Sq Rel
- δ < 1.25, 1.25², 1.25³

## Notes

- Keep input normalization and image resizing consistent between training and inference.
- If checkpoint keys differ, adapt `load_state_dict` calls in model class loaders.

## Future Work

- Add `docker` or `conda` environment setup.
- Add inference API (`FastAPI` / `uvicorn`).
- Add additional datasets (NYU Depth V2, Make3D).
- Add self-supervised pretraining and scale-aware loss.

## License

MIT License

## Author

- Project name: Monocular Depth Estimation
- Created by: [Your Name]
