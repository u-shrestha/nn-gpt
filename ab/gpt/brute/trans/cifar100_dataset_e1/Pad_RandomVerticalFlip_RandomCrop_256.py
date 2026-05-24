import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(135, 53, 187), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.44),
    transforms.RandomCrop(size=29),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
