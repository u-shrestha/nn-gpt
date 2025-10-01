import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.77),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 1.0), ratio=(1.29, 2.74)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
