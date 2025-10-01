import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.06, contrast=0.92, saturation=0.83, hue=0.04),
    transforms.RandomAffine(degrees=27, translate=(0.18, 0.16), scale=(0.97, 1.9), shear=(4.25, 8.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
