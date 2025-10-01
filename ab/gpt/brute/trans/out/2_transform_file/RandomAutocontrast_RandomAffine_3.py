import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.62),
    transforms.RandomAffine(degrees=6, translate=(0.2, 0.15), scale=(0.82, 1.43), shear=(2.78, 8.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
