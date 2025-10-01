import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=4, p=0.43),
    transforms.RandomAffine(degrees=27, translate=(0.05, 0.06), scale=(1.13, 1.68), shear=(3.17, 9.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
