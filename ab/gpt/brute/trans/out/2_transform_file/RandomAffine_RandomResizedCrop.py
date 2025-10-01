import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=23, translate=(0.15, 0.11), scale=(1.08, 1.88), shear=(3.64, 5.59)),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.93), ratio=(0.8, 1.78)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
