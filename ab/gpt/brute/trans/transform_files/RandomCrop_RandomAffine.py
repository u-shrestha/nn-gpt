import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(27, 41)),
    transforms.RandomAffine(degrees=16, translate=(0.14, 0.02), scale=(0.97, 0.84), shear=8.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
