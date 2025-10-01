import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.12),
    transforms.RandomAffine(degrees=24, translate=(0.02, 0.01), scale=(1.15, 1.49), shear=(3.41, 6.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
