import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.22),
    transforms.RandomAffine(degrees=5, translate=(0.17, 0.12), scale=(0.96, 1.42), shear=(1.77, 9.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
