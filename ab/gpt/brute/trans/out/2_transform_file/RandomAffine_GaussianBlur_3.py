import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.07, 0.0), scale=(1.14, 1.76), shear=(4.93, 5.53)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.87, 1.41)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
