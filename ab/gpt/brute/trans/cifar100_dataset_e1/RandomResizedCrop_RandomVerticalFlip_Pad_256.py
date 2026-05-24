import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.84), ratio=(1.32, 1.55)),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Pad(padding=5, fill=(137, 2, 32), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
