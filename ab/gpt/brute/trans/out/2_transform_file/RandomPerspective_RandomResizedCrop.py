import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.19, p=0.89),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.81), ratio=(1.28, 1.86)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
