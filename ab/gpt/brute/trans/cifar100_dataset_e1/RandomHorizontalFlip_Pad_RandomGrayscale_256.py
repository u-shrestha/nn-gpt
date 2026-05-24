import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.55),
    transforms.Pad(padding=1, fill=(59, 250, 137), padding_mode='edge'),
    transforms.RandomGrayscale(p=0.67),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
