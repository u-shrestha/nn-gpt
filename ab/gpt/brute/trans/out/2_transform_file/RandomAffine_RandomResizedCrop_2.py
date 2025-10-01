import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.08, 0.18), scale=(1.07, 1.44), shear=(4.66, 6.14)),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.82), ratio=(0.97, 1.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
