import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=7, translate=(0.06, 0.02), scale=(1.02, 1.86), shear=(2.91, 7.9)),
    transforms.RandomInvert(p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
