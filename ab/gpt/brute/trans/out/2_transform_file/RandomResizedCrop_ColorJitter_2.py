import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.83), ratio=(1.04, 1.74)),
    transforms.ColorJitter(brightness=0.93, contrast=1.14, saturation=0.91, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
