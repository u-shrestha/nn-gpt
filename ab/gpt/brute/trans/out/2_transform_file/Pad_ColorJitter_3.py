import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(13, 169, 57), padding_mode='constant'),
    transforms.ColorJitter(brightness=0.91, contrast=1.1, saturation=1.15, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
