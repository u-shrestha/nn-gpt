import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(57, 38)),
    transforms.RandomResizedCrop(size=60, scale=(0.68, 0.86), ratio=(0.78, 1.31)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
