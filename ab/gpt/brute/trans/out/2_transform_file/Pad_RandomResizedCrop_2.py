import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(66, 215, 199), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.95), ratio=(1.06, 2.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
