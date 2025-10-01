import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.77),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.88), ratio=(1.17, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
