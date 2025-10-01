import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=7, translate=(0.07, 0.12), scale=(0.85, 1.84), shear=(4.01, 7.27)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.06, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
