import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(175, 0, 2), padding_mode='constant'),
    transforms.RandomAffine(degrees=12, translate=(0.2, 0.01), scale=(1.03, 1.64), shear=(1.85, 8.86)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
