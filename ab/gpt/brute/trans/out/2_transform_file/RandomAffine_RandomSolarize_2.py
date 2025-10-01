import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=21, translate=(0.07, 0.15), scale=(1.13, 1.92), shear=(3.6, 8.91)),
    transforms.RandomSolarize(threshold=67, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
