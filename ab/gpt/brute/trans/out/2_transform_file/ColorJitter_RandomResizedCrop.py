import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.05, saturation=1.2, hue=0.03),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.94), ratio=(0.98, 2.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
