import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.76),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.86), ratio=(0.99, 2.22)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
