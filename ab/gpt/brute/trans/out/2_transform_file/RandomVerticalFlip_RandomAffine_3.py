import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.36),
    transforms.RandomAffine(degrees=25, translate=(0.12, 0.17), scale=(0.81, 1.5), shear=(1.33, 5.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
