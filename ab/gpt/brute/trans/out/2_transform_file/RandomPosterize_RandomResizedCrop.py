import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.37),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.97), ratio=(0.86, 2.69)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
