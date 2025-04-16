# Image Segmentation Assignment

This repository contains code for a two-part image segmentation assignment. The project is divided into:

- **Task 1**: Dataset Preparation using Python
- **Task 2**: Train an Image Segmentation Model using PyTorch Lightning

Both tasks have been implemented with a focus on clarity, handling edge cases, and reproducibility. The code is designed to run on a macOS (Apple Silicon) system with a Conda setup and supports both CPU and GPU execution in PyTorch Lightning.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Task 1: Dataset Preparation](#task-1-dataset-preparation)
  - [Script: `prepare_dataset.py`](#script-preparedatasetpy)
  - [Edge Cases Handled](#edge-cases-handled)
  - [Visualization of Masks](#visualization-of-masks)
- [Task 2: Model Training](#task-2-model-training)
- [Usage](#usage)
- [Reproducibility](#reproducibility)

---

## Overview

The objective of this project is to:
1. Process a small subset of the COCO dataset (e.g., COCO 2017 val) to generate segmentation masks.
2. Train an image segmentation model using PyTorch Lightning that can run on both CPU and GPU.

We have implemented a robust dataset preparation pipeline to handle common edge cases in COCO annotations and integrated a visualization utility to confirm mask generation.

---

## Requirements

The following libraries are required:
- Python (3.8+ recommended)
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- pycocotools (installed manually for macOS Apple Silicon)
- opencv-python
- numpy
- matplotlib
- Pillow
- tqdm
- torch, torchvision, torchaudio
- pytorch-lightning

---

## Installation

### 1. Create and Activate a Conda Environment

```bash
conda create -n coco-seg1 python=3.9 -y
conda activate coco-seg1

```

### 2. Install `uv` for Dependency Management

```bash
pip install uv
```

### 3. Install `pycocotools` and dependencies

```bash
uv pip install numpy pillow tqdm matplotlib requests pycocotools

```

### 4. Install Remaining Dependencies with `uv`

```bash
uv pip compile opencv-python numpy matplotlib Pillow tqdm torch torchvision torchaudio pytorch-lightning
```

---

## Task 1: Dataset Preparation

### Script: `prepare_dataset.py`

This script processes the COCO 2017 dataset to generate segmentation masks, handling overlapping masks, missing images, and invalid annotations. The script accepts images and annotations as input, creates masks, and saves them in an output directory.

#### Usage:

```bash
python prepare_dataset.py \
  --ann data/coco/annotations/instances_val2017.json \
  --img_dir data/coco/val2017 \
  --out_dir data/coco/val2017_masks \
  --visualize 3
```

#### Key Arguments:
- `--ann`: Path to the annotation JSON file.
- `--img_dir`: Directory containing the input images.
- `--out_dir`: Directory to save the generated masks.
- `--visualize`: (Optional) Number of random samples to visualize after processing.

### Edge Cases Handled

- **Missing or corrupted images:** Skips images if not found on disk.
- **No annotations for an image:** If no annotations are found, the image is skipped with a notice.
- **Overlapping masks:**
    - Overlapping masks of the same class: 
        - For semantic segmentation, a union of all overlapping regions of the same class is done.
        - For instance segmentation, track each object individually 
    - Overlapping masks of different classes: Handled using a hybrid priority-area strategy where larger objects and higher-priority classes retain their regions, smaller or lower-priority ones are clipped.

You want a union of all overlapping regions of the same class.
- **Crowd annotations (`iscrowd=1`):** Such annotations are ignored to avoid ambiguous masks.
- **Invalid or empty masks:** Annotations resulting in empty masks are not saved.



### Visualization of Masks

After mask generation, if the `--visualize` flag is provided, the script will display random samples showing:
- Original image
- Segmentation mask (with a colormap)
- Overlay of the mask on the original image

This enables quick verification that the mask generation process is correct.

---

## Task 2: Model Training

### Overview

Task 2 involves training an image segmentation model using PyTorch Lightning. The model can be switched to GPU mode if desired using a runtime flag in your training script.

### Key Points:
- **Model Architecture:** script runs for fastscnn, deeplabv3 and u-net
- **Metrics:** Model performance is evaluated using IoU, Dice Coefficient, and pixel accuracy.
- **Monitoring:** Training metrics are tracked using TensorBoard 

(Please refer to the separate files for detailed model implementation and training scripts.)

---

## Usage

<!-- ### Data Preparation

1. you must execute task 1 first
``` -->

Install dependencies 

```bash
uv pip install torch torchvision pillow numpy segmentation-models-pytorch tensorboard

```

Just run the following to automatically download, extract, and process the COCO dataset:

```bash
python prepare_dataset.py
```



### Model Training and inference

```bash
python train.py
```
---
Absolutely, here's a succinct, copy-paste-ready section for your `README.md`:

---

##  Training Script: `train.py`

Run the training with configurable options:

```bash
python train.py --data_root /path/to/coco --models unet deeplabv3 --gpu 0 --lr 1e-4 --batch_size 8 --epochs 25
```

### 🔧 Arguments

- `--data_root`: Path to dataset root (default: `/scratch/sanjotst/vjtech/data/coco`)
- `--models`: Space-separated list of models to train. Options: `unet`, `deeplabv3`, `fastscnn`
- `--gpu`: GPU index to use (default: `0`)
- `--lr`: Learning rate (default: `1e-6`)
- `--batch_size`: Batch size (default: `8`)
- `--epochs`: Number of training epochs (default: `20`)

You can train multiple models in one run by passing them together via `--models`.

--- 
the file structure is as 
runs/
└── {model}_{metric}_{timestamp}/
    ├── train/               # TensorBoard logs for training
    ├── val/                 # TensorBoard logs for validation
    ├── models/              # Saved model checkpoints (.pth)
    ├── metrics_first_iter.txt  # Metrics after first iteration
    └── final_metrics.txt       # Metrics after final epoch
.

---

