import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.83, contrast=0.83, saturation=0.92, hue=0.06),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.91), ratio=(1.27, 1.92)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
