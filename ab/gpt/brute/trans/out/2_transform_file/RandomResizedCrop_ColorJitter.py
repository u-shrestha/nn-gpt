import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.94), ratio=(1.03, 1.69)),
    transforms.ColorJitter(brightness=0.82, contrast=0.81, saturation=0.95, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
