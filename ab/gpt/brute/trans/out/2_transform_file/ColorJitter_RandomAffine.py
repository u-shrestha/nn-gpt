import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.06, contrast=0.99, saturation=1.15, hue=0.03),
    transforms.RandomAffine(degrees=20, translate=(0.08, 0.05), scale=(0.87, 1.84), shear=(4.78, 6.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
