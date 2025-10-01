import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.22, 1.33)),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.97), ratio=(1.16, 1.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
