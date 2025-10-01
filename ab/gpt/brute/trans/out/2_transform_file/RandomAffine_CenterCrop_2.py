import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.01, 0.14), scale=(1.0, 1.57), shear=(2.82, 6.32)),
    transforms.CenterCrop(size=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
