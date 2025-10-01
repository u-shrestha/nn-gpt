import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=23, translate=(0.08, 0.02), scale=(1.18, 1.6), shear=(0.2, 7.41)),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
