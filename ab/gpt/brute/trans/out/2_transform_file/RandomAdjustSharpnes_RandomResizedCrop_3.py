import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.87, p=0.2),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.86), ratio=(1.21, 2.29)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
