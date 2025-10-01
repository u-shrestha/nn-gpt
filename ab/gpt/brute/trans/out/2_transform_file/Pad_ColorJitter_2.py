import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(172, 40, 144), padding_mode='edge'),
    transforms.ColorJitter(brightness=1.2, contrast=0.84, saturation=1.12, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
