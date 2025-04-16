# Importing necessary libraries
import os
import numpy as np
from pycocotools.coco import COCO  # For reading COCO annotations
from PIL import Image  # For image manipulation
from tqdm import tqdm  # For progress bars
import argparse  # For command-line argument parsing
import matplotlib.pyplot as plt  # For visualization
import random  # For sampling
import zipfile  # For extracting zip files
import requests  # For downloading files
from pathlib import Path  # For convenient path manipulations

# Define a priority list of COCO categories for mask generation
# Categories earlier in the list have higher priority when overlapping
priority_classnames = [
    "person", "car", "bus", "truck", "bicycle", "motorcycle",
    "traffic light", "stop sign", "dog", "cat", "bird"
]

def download_and_extract_coco(root_dir="data/coco"):
    """
    Downloads and extracts COCO val2017 images and annotations.

    Args:
        root_dir (str): Directory to save dataset.

    Returns:
        dict: Paths to images, annotations, and output mask folder.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # URLs of the dataset files to download
    files = {
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    for filename, url in files.items():
        file_path = root / filename
        # Download only if file does not exist
        if not file_path.exists():
            print(f"Downloading {filename}...")
            r = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract files if not already extracted
        extract_dir = root / filename.replace(".zip", "")
        if not extract_dir.exists():
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(root)

    print("✅ COCO dataset downloaded and extracted.")
    return {
        "img_dir": str(root / "val2017"),
        "ann_file": str(root / "annotations" / "instances_val2017.json"),
        "mask_dir": str(root / "val2017_masks")
    }

def visualize_masks(image_dir, mask_dir, sample_count=3):
    """
    Visualizes sample masks by overlaying them on original images.

    Args:
        image_dir (str): Path to COCO images.
        mask_dir (str): Path to generated masks.
        sample_count (int): Number of random images to visualize.
    """
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    samples = random.sample(mask_files, min(sample_count, len(mask_files)))

    for mask_file in samples:
        img_file = mask_file.replace(".png", ".jpg")
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Skip if corresponding image is not found
        if not os.path.exists(img_path):
            print(f"Skipping {img_file}, not found in image dir.")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # Ensure grayscale

        unique_classes = np.unique(mask)
        # Generate random colors for each class (excluding background)
        colormap = {c: [random.randint(0,255) for _ in range(3)] for c in unique_classes if c != 0}
        
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Paint each class with its unique color
        for class_id, color in colormap.items():
            mask_indices = (mask == class_id)
            for c in range(3):
                color_mask[:, :, c][mask_indices] = color[c]

        # Overlay original image with color mask
        overlay = (0.6 * image + 0.4 * color_mask).astype(np.uint8)

        # Display original, mask, and overlay
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(mask, cmap="tab20")
        axs[1].set_title("Segmentation Mask")
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"visualization_{mask_file.replace('.png', '')}.png")
        plt.close()

def colorize_mask(mask):
    """
    Convert grayscale segmentation mask to RGB image using colormap.

    Args:
        mask (ndarray): 2D array where pixel value = category id.

    Returns:
        PIL.Image: RGB visualization of the mask.
    """
    colors = plt.cm.get_cmap('tab20', 256)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for label in np.unique(mask):
        if label == 0:
            continue  # background
        color = (np.array(colors(label)[:3]) * 255).astype(np.uint8)
        color_mask[mask == label] = color

    return Image.fromarray(color_mask)

def create_masks(annotation_file, image_dir, output_dir, classes=None):
    """
    Generates segmentation masks from COCO annotations.

    Args:
        annotation_file (str): Path to COCO annotation JSON.
        image_dir (str): Directory with COCO images.
        output_dir (str): Directory to save generated masks.
        classes (list[str], optional): List of class names to include.
    """
    coco = COCO(annotation_file)
    os.makedirs(output_dir, exist_ok=True)

    # Get category IDs for selected class names
    cat_ids = coco.getCatIds(catNms=classes) if classes else coco.getCatIds()
    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"Found {len(cat_ids)} categories and {len(img_ids)} images")

    for img_id in tqdm(img_ids, desc="Generating masks"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids))
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Load category information for priority handling
        cat_info = coco.loadCats(coco.getCatIds())
        cat_name_to_id = {cat['name']: cat['id'] for cat in cat_info}
        
        # Build priority dictionary based on predefined order
        category_priority = {
            cat_name_to_id[name]: i for i, name in enumerate(priority_classnames) if name in cat_name_to_id
        }

        # Sort annotations: priority first, then by area (largest first)
        anns.sort(key=lambda x: (
            category_priority.get(x["category_id"], 999),
            -x["area"]
        ))

        if not anns:
            print(f"Note: No annotations for image {img_info['file_name']}, skipping.")
            continue

        for ann in anns:
            cat_id = ann['category_id']
            mask_instance = coco.annToMask(ann)
            # Apply only if no previous label or same label already exists
            mask[(mask_instance == 1) & ((mask == 0) | (mask == cat_id))] = cat_id

        # Save colorized mask
        mask_img = colorize_mask(mask)
        mask_img.save(os.path.join(output_dir, f"{img_info['file_name'].split('.')[0]}.png"))

# Main function
if __name__ == "__main__":
    # Download dataset (if not already downloaded)
    default_paths = download_and_extract_coco()

    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default=default_paths["ann_file"], help="Path to annotation json")
    parser.add_argument("--img_dir", default=default_paths["img_dir"], help="Path to images")
    parser.add_argument("--out_dir", default=default_paths["mask_dir"], help="Output mask directory")
    parser.add_argument("--classes", default="person", nargs="+", help="List of class names (optional)")
    parser.add_argument("--visualize", type=int, default=3, help="Number of random samples to visualize")

    args = parser.parse_args()

    # Generate masks from annotations
    create_masks(args.ann, args.img_dir, args.out_dir, args.classes)

    # Optionally visualize some samples
    if int(args.visualize) > 0:
        visualize_masks(args.img_dir, args.out_dir, sample_count=int(args.visualize))
