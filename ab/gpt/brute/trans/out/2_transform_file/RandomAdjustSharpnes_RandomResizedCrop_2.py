import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.11, p=0.75),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.92), ratio=(0.97, 2.11)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
