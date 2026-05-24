import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(223, 160, 89), padding_mode='constant'),
    transforms.RandomCrop(size=24),
    transforms.RandomVerticalFlip(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
