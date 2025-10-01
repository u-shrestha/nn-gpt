import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=8, translate=(0.05, 0.11), scale=(0.92, 1.84), shear=(2.68, 9.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
