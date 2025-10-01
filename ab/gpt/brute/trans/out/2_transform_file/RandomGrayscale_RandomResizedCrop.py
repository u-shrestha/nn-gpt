import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.69),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.96), ratio=(1.2, 2.65)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
