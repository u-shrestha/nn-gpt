import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.14, p=0.73),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.99), ratio=(1.1, 1.76)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
