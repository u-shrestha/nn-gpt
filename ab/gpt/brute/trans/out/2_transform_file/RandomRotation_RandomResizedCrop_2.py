import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=4),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.98), ratio=(0.86, 2.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
