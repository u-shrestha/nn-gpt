import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.43),
    transforms.RandomAffine(degrees=27, translate=(0.07, 0.01), scale=(0.97, 1.21), shear=(3.66, 7.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
