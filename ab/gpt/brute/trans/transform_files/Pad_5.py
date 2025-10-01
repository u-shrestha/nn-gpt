import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(96, 73, 167), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
