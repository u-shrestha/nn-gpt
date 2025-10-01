import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.17, 0.03), scale=(1.13, 1.88), shear=(0.07, 8.49)),
    transforms.RandomVerticalFlip(p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
