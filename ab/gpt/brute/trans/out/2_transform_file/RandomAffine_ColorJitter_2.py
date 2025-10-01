import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=21, translate=(0.16, 0.04), scale=(0.9, 1.37), shear=(2.38, 6.48)),
    transforms.ColorJitter(brightness=1.18, contrast=1.14, saturation=0.91, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
