import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.82), ratio=(0.84, 1.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
