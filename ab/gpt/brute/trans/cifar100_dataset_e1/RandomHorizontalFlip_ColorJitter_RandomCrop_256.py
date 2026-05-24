import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.ColorJitter(brightness=1.18, contrast=0.84, saturation=1.13, hue=0.02),
    transforms.RandomCrop(size=25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
