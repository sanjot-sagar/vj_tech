

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random

class PersonSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Binary mask: person (255) vs background (0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()  # Convert to binary mask (0 or 1)

        return image, mask

def get_dataloaders(root_dir, batch_size=8, num_workers=4):
    image_dir = os.path.join(root_dir, 'val2017')
    mask_dir = os.path.join(root_dir, 'val2017_masks')
    
    # Get the list of mask files that actually exist
    mask_files = sorted(os.listdir(mask_dir))
    
    # Create pairs of image and mask paths where masks actually exist
    valid_pairs = []
    for mask_file in mask_files:
        img_file = mask_file.replace('.png', '.jpg')
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Only add if the image also exists
        if os.path.isfile(img_path):
            valid_pairs.append((img_path, mask_path))
    
    # Shuffle the pairs with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(valid_pairs)
    
    # Split the pairs
    total = len(valid_pairs)
    train_end = int(0.8 * total)
    val_end = int(0.95 * total)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    # Unzip the pairs
    train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    val_images, val_masks = zip(*val_pairs) if val_pairs else ([], [])
    test_images, test_masks = zip(*test_pairs) if test_pairs else ([], [])

    # Transforms
    common_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset objects
    train_set = PersonSegmentationDataset(train_images, train_masks, transform=common_transform)
    val_set = PersonSegmentationDataset(val_images, val_masks, transform=common_transform)
    test_set = PersonSegmentationDataset(test_images, test_masks, transform=common_transform)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print dataset sizes for debugging
    print(f"Total valid image-mask pairs: {total}")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Testing samples: {len(test_images)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root = "/scratch/sanjotst/vjtech/data/coco"
    train_loader, val_loader, test_loader = get_dataloaders(root)

    # Debug print
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
