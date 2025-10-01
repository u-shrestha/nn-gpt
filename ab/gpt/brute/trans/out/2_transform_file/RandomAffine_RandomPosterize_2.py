import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.01, 0.06), scale=(0.88, 1.75), shear=(0.51, 8.15)),
    transforms.RandomPosterize(bits=7, p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
