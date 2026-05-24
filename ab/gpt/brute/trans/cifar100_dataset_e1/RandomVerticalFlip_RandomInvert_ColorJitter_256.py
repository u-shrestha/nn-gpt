import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.77),
    transforms.RandomInvert(p=0.27),
    transforms.ColorJitter(brightness=1.18, contrast=1.05, saturation=1.08, hue=0.03),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
