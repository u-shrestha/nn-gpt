import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.96), ratio=(1.14, 2.85)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.83, 1.76)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
