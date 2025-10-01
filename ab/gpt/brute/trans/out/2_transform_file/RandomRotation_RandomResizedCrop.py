import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=11),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.89), ratio=(1.27, 2.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
