import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.99), ratio=(1.2, 1.65)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
