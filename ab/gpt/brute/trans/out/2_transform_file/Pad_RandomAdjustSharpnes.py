import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(239, 68, 29), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.9, p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
