import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=1.2, saturation=1.07, hue=0.08),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.92), ratio=(0.89, 2.64)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
