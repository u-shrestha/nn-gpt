import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.07, 0.14), scale=(0.91, 1.49), shear=(1.01, 6.27)),
    transforms.RandomSolarize(threshold=84, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
