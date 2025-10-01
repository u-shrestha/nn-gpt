import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.88), ratio=(1.04, 2.43)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.76, 1.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
