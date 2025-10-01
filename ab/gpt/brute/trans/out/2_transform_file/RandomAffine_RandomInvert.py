import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.01, 0.04), scale=(1.0, 1.95), shear=(1.32, 9.14)),
    transforms.RandomInvert(p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
