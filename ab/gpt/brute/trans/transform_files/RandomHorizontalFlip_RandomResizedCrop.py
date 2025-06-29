import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.48),
    transforms.RandomResizedCrop(size=42, scale=(0.76, 0.88), ratio=(0.78, 1.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
