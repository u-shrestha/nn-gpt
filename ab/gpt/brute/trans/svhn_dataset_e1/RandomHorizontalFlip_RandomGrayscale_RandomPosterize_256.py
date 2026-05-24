import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.32),
    transforms.RandomGrayscale(p=0.76),
    transforms.RandomPosterize(bits=8, p=0.16),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
