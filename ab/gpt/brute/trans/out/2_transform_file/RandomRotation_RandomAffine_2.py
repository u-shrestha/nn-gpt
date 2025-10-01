import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=1),
    transforms.RandomAffine(degrees=26, translate=(0.11, 0.16), scale=(1.07, 1.68), shear=(0.21, 6.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
