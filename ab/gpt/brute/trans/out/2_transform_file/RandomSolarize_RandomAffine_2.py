import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=36, p=0.34),
    transforms.RandomAffine(degrees=4, translate=(0.2, 0.19), scale=(1.16, 1.29), shear=(2.43, 5.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
