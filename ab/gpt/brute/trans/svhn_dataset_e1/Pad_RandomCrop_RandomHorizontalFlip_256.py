import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(162, 51, 17), padding_mode='edge'),
    transforms.RandomCrop(size=31),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
