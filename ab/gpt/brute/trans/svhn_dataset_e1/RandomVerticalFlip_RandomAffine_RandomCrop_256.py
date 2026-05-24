import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.38),
    transforms.RandomAffine(degrees=17, translate=(0.13, 0.15), scale=(1.11, 1.39), shear=(2.74, 5.44)),
    transforms.RandomCrop(size=30),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
