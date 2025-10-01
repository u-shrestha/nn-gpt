import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.93), ratio=(0.86, 2.46)),
    transforms.RandomAffine(degrees=29, translate=(0.04, 0.05), scale=(1.13, 1.4), shear=(3.72, 7.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
