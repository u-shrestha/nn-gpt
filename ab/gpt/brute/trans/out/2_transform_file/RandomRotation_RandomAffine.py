import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=24),
    transforms.RandomAffine(degrees=19, translate=(0.07, 0.01), scale=(1.12, 1.43), shear=(3.47, 7.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
