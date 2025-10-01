import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=29, translate=(0.13, 0.15), scale=(1.02, 1.91), shear=(1.25, 9.28)),
    transforms.RandomAutocontrast(p=0.41),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
