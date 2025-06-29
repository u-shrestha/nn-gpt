import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.58),
    transforms.ColorJitter(brightness=1.16, contrast=1.11, saturation=1.0, hue=-0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
