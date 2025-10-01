import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.13, 0.11), scale=(0.85, 1.44), shear=(2.07, 8.49)),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.96), ratio=(1.25, 2.22)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
