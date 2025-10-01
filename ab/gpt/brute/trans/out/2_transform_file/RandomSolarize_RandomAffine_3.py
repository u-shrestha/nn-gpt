import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=250, p=0.84),
    transforms.RandomAffine(degrees=26, translate=(0.18, 0.15), scale=(0.97, 1.73), shear=(4.45, 7.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
