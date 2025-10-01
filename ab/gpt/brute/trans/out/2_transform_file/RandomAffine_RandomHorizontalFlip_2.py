import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=6, translate=(0.18, 0.16), scale=(0.97, 1.32), shear=(1.74, 8.23)),
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
