import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=24, translate=(0.04, 0.14), scale=(0.87, 1.73), shear=(2.26, 9.06)),
    transforms.RandomInvert(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
