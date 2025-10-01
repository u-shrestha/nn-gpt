import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(148, 118, 71), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.88), ratio=(0.92, 2.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
