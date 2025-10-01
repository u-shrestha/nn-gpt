import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.08, 0.19), scale=(0.91, 1.28), shear=(4.64, 5.29)),
    transforms.RandomGrayscale(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
