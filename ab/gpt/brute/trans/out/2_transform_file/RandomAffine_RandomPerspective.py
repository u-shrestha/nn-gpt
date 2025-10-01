import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=28, translate=(0.09, 0.08), scale=(1.01, 1.45), shear=(0.71, 9.17)),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.47),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
