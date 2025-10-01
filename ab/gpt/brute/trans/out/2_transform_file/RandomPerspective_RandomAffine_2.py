import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.14),
    transforms.RandomAffine(degrees=14, translate=(0.17, 0.07), scale=(0.96, 1.37), shear=(4.05, 9.36)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
