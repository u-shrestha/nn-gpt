import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.34),
    transforms.RandomAffine(degrees=0, translate=(0.19, 0.17), scale=(1.18, 1.2), shear=(2.86, 6.62)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
