import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.28, 1.38)),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.98), ratio=(1.32, 2.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
