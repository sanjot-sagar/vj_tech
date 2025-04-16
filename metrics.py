import torch
import torch.nn.functional as F

def dice_score(preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def iou_score(preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()

def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    correct = (preds == targets).sum()
    total = torch.numel(preds)
    return (correct / total).item()


# --------- Testing with dummy tensors ---------
if __name__ == "__main__":
    # Simulate batch of 2 predictions and targets (binary masks)
    preds = torch.tensor([
        [[[0.1, 0.9], [0.8, 0.2]]],
        [[[0.6, 0.4], [0.3, 0.7]]]
    ])  # shape: [2, 1, 2, 2]

    targets = torch.tensor([
        [[[0, 1], [1, 0]]],
        [[[1, 0], [0, 1]]]
    ]).float()

    print(f"Dice Score:        {dice_score(preds, targets):.4f}")
    print(f"IoU Score:         {iou_score(preds, targets):.4f}")
    print(f"Pixel Accuracy:    {pixel_accuracy(preds, targets):.4f}")
