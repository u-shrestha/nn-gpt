import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.03, 0.04), scale=(1.1, 1.87), shear=(2.1, 6.37)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.49, 1.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
