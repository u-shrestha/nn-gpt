import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=1.17, saturation=1.12, hue=0.01),
    transforms.RandomAffine(degrees=27, translate=(0.03, 0.15), scale=(1.05, 1.87), shear=(3.7, 9.45)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
