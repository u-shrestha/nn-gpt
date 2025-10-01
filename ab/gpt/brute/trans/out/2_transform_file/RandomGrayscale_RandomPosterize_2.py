import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.48),
    transforms.RandomPosterize(bits=6, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
