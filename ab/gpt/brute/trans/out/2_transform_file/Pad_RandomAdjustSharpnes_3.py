import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(201, 173, 83), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.89, p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
