import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.11, contrast=1.02, saturation=0.95, hue=0.02),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.49, 1.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
