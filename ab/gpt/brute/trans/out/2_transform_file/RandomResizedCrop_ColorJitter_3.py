import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.86), ratio=(0.8, 2.56)),
    transforms.ColorJitter(brightness=0.93, contrast=0.95, saturation=1.13, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
