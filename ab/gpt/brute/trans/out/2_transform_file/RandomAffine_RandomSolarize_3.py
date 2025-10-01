import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.14, 0.14), scale=(0.83, 1.41), shear=(1.78, 7.03)),
    transforms.RandomSolarize(threshold=7, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
