import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.86), ratio=(0.95, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
