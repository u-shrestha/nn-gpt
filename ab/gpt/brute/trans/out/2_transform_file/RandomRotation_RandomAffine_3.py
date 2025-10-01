import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=29, translate=(0.15, 0.07), scale=(0.9, 1.51), shear=(0.48, 8.84)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
