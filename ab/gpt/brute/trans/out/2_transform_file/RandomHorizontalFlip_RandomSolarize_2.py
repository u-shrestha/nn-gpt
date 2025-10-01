import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.RandomSolarize(threshold=238, p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
