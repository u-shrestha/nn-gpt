import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomCrop(size=28),
    transforms.ColorJitter(brightness=1.15, contrast=1.02, saturation=1.04, hue=0.09),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
