import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.2)),
    transforms.RandomAffine(degrees=15, translate=(0.11, 0.02), scale=(0.94, 1.45), shear=(1.39, 8.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
