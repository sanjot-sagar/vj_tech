import argparse
import os
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from models import get_model
from metrics import dice_score, iou_score, pixel_accuracy

# --------------- Argument Parsing -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Segmentation Models")
    parser.add_argument('--data_root', type=str, default="/scratch/sanjotst/vjtech/data/coco", help='Path to the dataset root directory')
    parser.add_argument('--models', nargs='+', default=[ 'unet', 'deeplabv3', 'fastscnn'],
                        help='List of models to train [unet, deeplabv3, fastscnn]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (0-7)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')

    return parser.parse_args()

# --------------- Directory Utils -------------------
def create_log_dirs(base_dir, model_name, metric):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = Path(base_dir) / f"{model_name}_{metric}_{timestamp}"
    train_log_dir = log_base / 'train'
    val_log_dir = log_base / 'val'
    train_log_dir.mkdir(parents=True, exist_ok=True)
    val_log_dir.mkdir(parents=True, exist_ok=True)
    return train_log_dir, val_log_dir, log_base

def save_model(model, log_base, name, epoch):
    model_dir = log_base / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}_epoch{epoch}.pth"
    torch.save(model.state_dict(), path)

# --------------- Evaluation -------------------
def evaluate(model, dataloader, device):
    model.eval()
    dice_scores, iou_scores, pixel_accs = [], [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Extract the prediction tensor from the OrderedDict for DeepLabV3
            # Extract the prediction tensor for torchvision models
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)

            dice_scores.append(dice)
            iou_scores.append(iou)
            pixel_accs.append(acc)

    return {
        'dice': sum(dice_scores) / len(dice_scores),
        'iou': sum(iou_scores) / len(iou_scores),
        'pixel_accuracy': sum(pixel_accs) / len(pixel_accs)
    }

# --------------- Training Loop -------------------
def train_one_model(model_name, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(args.data_root, args.batch_size)
    model = get_model(model_name).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_log_dir, val_log_dir, log_base = create_log_dirs("runs", model_name, "seg")
    train_writer = SummaryWriter(log_dir=train_log_dir)
    val_writer = SummaryWriter(log_dir=val_log_dir)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            # Extract the prediction tensor for torchvision models
            # if output is is dict (DeepLabV3)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if epoch == 0 and batch_idx == 0:
                # Save model checkpoint + inference
                save_model(model, log_base, model_name, epoch=0)
                metrics = evaluate(model, test_loader, device)
                # Save metrics to file
                metrics_file = log_base / f"metrics_first_iter.txt"
                with open(metrics_file, 'w') as f:
                    for k, v in metrics.items():
                        f.write(f"{k}: {v:.4f}\n")

        avg_loss = running_loss / len(train_loader)
        train_writer.add_scalar('Loss', avg_loss, epoch)

        # Evaluation
        val_metrics = evaluate(model, val_loader, device)
        train_metrics = evaluate(model, train_loader, device)

        for metric in ['dice', 'iou', 'pixel_accuracy']:
            train_writer.add_scalar(metric, train_metrics[metric], epoch)
            val_writer.add_scalar(metric, val_metrics[metric], epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f} | Val Dice: {val_metrics['dice']:.4f}")

        # Save model at the end
        if epoch + 1 == args.epochs:
            save_model(model, log_base, model_name, epoch=epoch+1)
            final_metrics = evaluate(model, test_loader, device)
            metrics_file = log_base / f"final_metrics.txt"
            with open(metrics_file, 'w') as f:
                for k, v in final_metrics.items():
                    f.write(f"{k}: {v:.4f}\n")

# --------------- Main -------------------
if __name__ == '__main__':
    args = parse_args()

    for model_name in args.models:
        print(f"\n==== Training {model_name} ====")
        train_one_model(model_name, args)
