import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=239, p=0.57),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.97), ratio=(1.2, 2.63)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
