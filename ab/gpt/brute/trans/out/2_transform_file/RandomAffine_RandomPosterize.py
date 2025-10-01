import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.04, 0.09), scale=(0.85, 1.62), shear=(0.41, 7.66)),
    transforms.RandomPosterize(bits=4, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
