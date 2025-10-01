import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=1.07, saturation=0.96, hue=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=1.21, p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
