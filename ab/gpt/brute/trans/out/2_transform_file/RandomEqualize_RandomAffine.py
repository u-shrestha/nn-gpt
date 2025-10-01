import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.6),
    transforms.RandomAffine(degrees=7, translate=(0.06, 0.19), scale=(1.02, 1.33), shear=(2.64, 6.09)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
