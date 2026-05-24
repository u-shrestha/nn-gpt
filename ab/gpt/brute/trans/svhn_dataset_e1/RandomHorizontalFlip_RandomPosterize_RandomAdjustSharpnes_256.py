import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.54),
    transforms.RandomPosterize(bits=7, p=0.12),
    transforms.RandomAdjustSharpness(sharpness_factor=1.77, p=0.57),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
