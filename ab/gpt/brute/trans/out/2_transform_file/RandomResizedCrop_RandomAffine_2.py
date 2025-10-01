import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.91), ratio=(1.06, 2.44)),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.17), scale=(1.02, 1.67), shear=(2.65, 9.82)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
