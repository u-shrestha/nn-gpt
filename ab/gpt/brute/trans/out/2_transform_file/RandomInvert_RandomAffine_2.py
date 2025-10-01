import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.78),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.19), scale=(0.8, 1.61), shear=(4.77, 7.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
