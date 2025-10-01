import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=1.17, saturation=1.06, hue=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=1.44, p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
