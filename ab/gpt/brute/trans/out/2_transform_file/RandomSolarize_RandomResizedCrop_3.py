import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=150, p=0.71),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.84), ratio=(1.22, 3.0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
