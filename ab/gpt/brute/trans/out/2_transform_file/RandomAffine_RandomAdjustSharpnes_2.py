import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=9, translate=(0.05, 0.14), scale=(1.04, 1.71), shear=(4.78, 5.98)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.81, p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
