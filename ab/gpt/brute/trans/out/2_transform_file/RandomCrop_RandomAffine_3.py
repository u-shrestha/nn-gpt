import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomAffine(degrees=4, translate=(0.09, 0.08), scale=(0.93, 1.29), shear=(0.58, 9.31)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
