import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.73),
    transforms.RandomAffine(degrees=23, translate=(0.16, 0.07), scale=(0.92, 1.53), shear=(0.57, 6.66)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
