import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=21, translate=(0.04, 0.04), scale=(1.08, 1.93), shear=(0.7, 5.36)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.17, 1.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
