import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.17, 0.01), scale=(1.04, 1.31), shear=(2.01, 7.77)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.72, p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
