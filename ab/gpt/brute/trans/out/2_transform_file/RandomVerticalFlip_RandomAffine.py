import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.12),
    transforms.RandomAffine(degrees=8, translate=(0.03, 0.14), scale=(0.93, 1.85), shear=(2.34, 6.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
