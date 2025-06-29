import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomAutocontrast(p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
