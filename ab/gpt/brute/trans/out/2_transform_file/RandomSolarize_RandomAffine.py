import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=153, p=0.18),
    transforms.RandomAffine(degrees=18, translate=(0.2, 0.05), scale=(1.18, 1.91), shear=(0.54, 5.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
