import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.18, 1.76)),
    transforms.CenterCrop(size=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
