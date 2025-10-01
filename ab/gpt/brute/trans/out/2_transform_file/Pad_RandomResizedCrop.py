import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(56, 73, 227), padding_mode='symmetric'),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.93), ratio=(1.26, 2.22)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
