import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=3, translate=(0.0, 0.07), scale=(1.17, 1.94), shear=(3.29, 8.82)),
    transforms.RandomEqualize(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
