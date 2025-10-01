import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.71, 1.44)),
    transforms.RandomAffine(degrees=10, translate=(0.19, 0.12), scale=(1.05, 1.97), shear=(4.96, 8.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
