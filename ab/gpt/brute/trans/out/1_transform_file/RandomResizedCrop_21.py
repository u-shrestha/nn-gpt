import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.9), ratio=(1.24, 2.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
