import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.91), ratio=(0.89, 1.69)),
    transforms.RandomAffine(degrees=8, translate=(0.18, 0.07), scale=(1.12, 1.34), shear=(3.1, 7.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
