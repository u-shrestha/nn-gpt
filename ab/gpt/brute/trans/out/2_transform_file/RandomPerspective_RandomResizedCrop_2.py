import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.13, p=0.87),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.85), ratio=(1.28, 1.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
