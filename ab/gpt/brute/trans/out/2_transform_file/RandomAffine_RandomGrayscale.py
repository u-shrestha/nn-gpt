import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=19, translate=(0.01, 0.09), scale=(0.82, 1.73), shear=(0.25, 6.57)),
    transforms.RandomGrayscale(p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
