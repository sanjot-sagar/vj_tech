import torch
import torch.nn as nn
import torchvision.models.segmentation as tv_models

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")

def get_model(model_name: str, num_classes: int = 1):
    model_name = model_name.lower()

    if model_name == 'unet':
        model = smp.Unet(
            encoder_name="resnet18",  # lightweight encoder
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif model_name == 'deeplabv3':
        model = tv_models.deeplabv3_mobilenet_v3_large(pretrained=True)
        # Replace classifier to output 1 class
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif model_name == 'fastscnn':
        model = FastSCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from [unet, deeplabv3, fastscnn]")

    return model

# ---------- Fast-SCNN lightweight implementation ----------
class FastSCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        return x

# ---------- Testing all models ----------
if __name__ == "__main__":
    model_names = ['unet', 'deeplabv3', 'fastscnn']
    x = torch.randn(2, 3, 256, 256)

    for name in model_names:
        print(f"Testing {name}...")
        model = get_model(name)
        y = model(x)
        if isinstance(y, dict):  # torchvision models return dict
            y = y['out']
        print(f"{name} output shape: {y.shape}")
