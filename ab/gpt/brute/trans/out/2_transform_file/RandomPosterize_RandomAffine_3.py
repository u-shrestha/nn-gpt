import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.78),
    transforms.RandomAffine(degrees=8, translate=(0.09, 0.1), scale=(0.95, 1.42), shear=(0.74, 9.08)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
