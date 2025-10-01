import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.18, 0.09), scale=(0.95, 1.51), shear=(0.06, 5.08)),
    transforms.ColorJitter(brightness=1.09, contrast=1.17, saturation=0.84, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
