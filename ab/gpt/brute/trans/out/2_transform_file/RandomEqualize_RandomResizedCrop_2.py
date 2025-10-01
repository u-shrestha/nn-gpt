import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.61),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.8), ratio=(0.94, 2.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
