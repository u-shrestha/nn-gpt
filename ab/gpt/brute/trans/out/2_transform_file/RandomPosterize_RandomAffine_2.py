import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=5, p=0.71),
    transforms.RandomAffine(degrees=30, translate=(0.04, 0.12), scale=(1.05, 1.51), shear=(1.43, 7.27)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
