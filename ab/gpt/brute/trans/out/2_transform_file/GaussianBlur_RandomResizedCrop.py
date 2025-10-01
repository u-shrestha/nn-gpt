import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.95, 1.29)),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.82), ratio=(0.9, 2.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
