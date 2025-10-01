import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.09, 0.15), scale=(1.0, 1.33), shear=(2.8, 5.25)),
    transforms.RandomHorizontalFlip(p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
