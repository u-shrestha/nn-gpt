import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(128, 3, 79), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.35, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
