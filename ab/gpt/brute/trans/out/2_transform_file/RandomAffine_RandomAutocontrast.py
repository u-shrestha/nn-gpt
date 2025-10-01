import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.03, 0.04), scale=(0.96, 1.74), shear=(1.99, 9.58)),
    transforms.RandomAutocontrast(p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
