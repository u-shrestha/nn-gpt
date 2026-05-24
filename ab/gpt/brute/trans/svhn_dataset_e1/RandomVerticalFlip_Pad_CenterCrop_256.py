import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.21),
    transforms.Pad(padding=4, fill=(185, 201, 13), padding_mode='edge'),
    transforms.CenterCrop(size=30),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
