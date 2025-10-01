import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.24, 1.94)),
    transforms.RandomAffine(degrees=22, translate=(0.16, 0.12), scale=(1.14, 1.24), shear=(2.1, 8.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
