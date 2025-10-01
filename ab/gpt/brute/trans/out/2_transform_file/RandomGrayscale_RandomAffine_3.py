import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.88),
    transforms.RandomAffine(degrees=7, translate=(0.13, 0.14), scale=(0.86, 1.99), shear=(2.33, 6.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
