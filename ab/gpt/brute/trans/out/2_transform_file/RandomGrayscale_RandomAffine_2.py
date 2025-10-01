import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.76),
    transforms.RandomAffine(degrees=22, translate=(0.19, 0.16), scale=(0.87, 1.84), shear=(1.91, 9.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
