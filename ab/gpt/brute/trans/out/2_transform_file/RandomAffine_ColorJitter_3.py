import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.17, 0.03), scale=(0.87, 1.39), shear=(1.42, 8.22)),
    transforms.ColorJitter(brightness=1.19, contrast=1.13, saturation=0.81, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
