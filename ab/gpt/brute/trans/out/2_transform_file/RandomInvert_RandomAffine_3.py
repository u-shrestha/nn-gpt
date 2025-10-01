import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.85),
    transforms.RandomAffine(degrees=0, translate=(0.09, 0.05), scale=(1.12, 1.35), shear=(0.75, 6.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
