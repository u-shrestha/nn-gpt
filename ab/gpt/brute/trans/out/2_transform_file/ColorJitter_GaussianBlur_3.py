import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.19, contrast=0.95, saturation=0.91, hue=0.07),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.44, 1.98)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
