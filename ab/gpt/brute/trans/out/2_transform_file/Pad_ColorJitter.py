import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(175, 204, 248), padding_mode='edge'),
    transforms.ColorJitter(brightness=0.95, contrast=1.05, saturation=0.88, hue=0.0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
