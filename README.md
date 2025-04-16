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
conda create -n coco-seg python=3.10
conda activate coco-seg
```

### 2. Install `uv` for Dependency Management

```bash
pip install uv
```

### 3. Install `pycocotools` Manually (for macOS Apple Silicon)

```bash
pip install cython
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

### 4. Install Remaining Dependencies with `uv`

```bash
uv pip compile opencv-python numpy matplotlib Pillow tqdm torch torchvision torchaudio pytorch-lightning
uv pip sync requirements.txt
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
        - For instance segmentation, track each object individually (TBD)
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
- **Model Architecture:** Choose a suitable architecture (e.g., UNet, DeepLabV3). Modify the training script for your needs.
- **Metrics:** Model performance is evaluated using IoU, Dice Coefficient, and pixel accuracy.
- **Monitoring:** Training metrics are tracked using TensorBoard or WandB.
- **Runtime Configuration:** By default, the code is set to run on CPU but includes a flag to switch to GPU.

(Please refer to the separate files for detailed model implementation and training scripts.)

---

## Usage

<!-- ### Data Preparation

1. **Download the COCO 2017 subset** and annotations:

```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
``` -->
### Data Preparation

Just run the following to automatically download, extract, and process the COCO dataset:

```bash
python prepare_dataset.py
```

**To customize inputs::**

```bash
python prepare_dataset.py \
  --ann data/coco/annotations/instances_val2017.json \
  --img_dir data/coco/val2017 \
  --out_dir data/coco/val2017_masks \
  --visualize 3
```

### Model Training

(Instructions for training the segmentation model using PyTorch Lightning go here. Ensure your training scripts handle both CPU and GPU modes.)

---

## Reproducibility

The project is set up to be reproducible:
- **Dependencies:** Managed via `uv` with a pinned `requirements.txt`.
- **Instructions:** Clear instructions are provided to download and process the dataset.
- **Environment:** Tested on macOS (Apple Silicon) using a Conda setup.

Please refer to each script’s documentation and inline comments for further details.

---

Enjoy working on your assignment!

