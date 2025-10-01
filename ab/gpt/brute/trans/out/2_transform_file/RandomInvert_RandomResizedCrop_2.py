import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.32),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.94), ratio=(1.03, 1.83)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
