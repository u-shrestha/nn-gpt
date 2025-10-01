import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.72),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.07), scale=(0.83, 1.59), shear=(1.67, 9.56)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
