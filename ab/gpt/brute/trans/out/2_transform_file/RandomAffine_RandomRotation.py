import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.11, 0.15), scale=(1.16, 1.43), shear=(0.88, 8.38)),
    transforms.RandomRotation(degrees=23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
