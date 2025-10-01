import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=23, p=0.14),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.95), ratio=(1.32, 1.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
