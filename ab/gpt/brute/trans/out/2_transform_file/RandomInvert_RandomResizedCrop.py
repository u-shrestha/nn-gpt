import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.42),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.92), ratio=(1.3, 1.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
