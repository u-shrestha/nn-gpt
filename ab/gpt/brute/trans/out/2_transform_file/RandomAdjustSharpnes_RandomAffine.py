import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.86, p=0.33),
    transforms.RandomAffine(degrees=6, translate=(0.15, 0.12), scale=(1.02, 1.63), shear=(0.06, 5.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
