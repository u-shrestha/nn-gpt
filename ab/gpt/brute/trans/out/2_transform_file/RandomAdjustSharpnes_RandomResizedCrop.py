import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.87, p=0.21),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.97), ratio=(0.83, 1.92)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
