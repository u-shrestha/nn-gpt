import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.02, 0.05), scale=(1.06, 1.28), shear=(2.16, 7.75)),
    transforms.RandomPosterize(bits=5, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
