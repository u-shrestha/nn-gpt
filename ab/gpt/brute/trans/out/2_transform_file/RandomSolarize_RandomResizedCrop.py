import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=231, p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.95), ratio=(1.04, 1.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
