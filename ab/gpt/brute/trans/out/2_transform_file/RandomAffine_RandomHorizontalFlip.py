import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.19, 0.12), scale=(1.06, 1.27), shear=(0.35, 6.98)),
    transforms.RandomHorizontalFlip(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
