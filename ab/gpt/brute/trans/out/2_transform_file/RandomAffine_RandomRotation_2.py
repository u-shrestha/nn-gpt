import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.17, 0.07), scale=(0.94, 1.21), shear=(3.53, 6.23)),
    transforms.RandomRotation(degrees=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
