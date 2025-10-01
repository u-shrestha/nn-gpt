import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.63, p=0.3),
    transforms.RandomAffine(degrees=16, translate=(0.12, 0.08), scale=(0.87, 1.21), shear=(2.16, 6.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
