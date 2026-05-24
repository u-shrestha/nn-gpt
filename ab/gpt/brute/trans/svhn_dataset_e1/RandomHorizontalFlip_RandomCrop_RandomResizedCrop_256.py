import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.RandomCrop(size=26),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.94), ratio=(1.12, 1.34)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
