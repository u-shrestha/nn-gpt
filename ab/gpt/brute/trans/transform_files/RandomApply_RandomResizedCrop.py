import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomApply(transforms=[RandomHorizontalFlip(p=0.5)], p=0.28),
    transforms.RandomResizedCrop(size=38, scale=(0.57, 0.96), ratio=(0.78, 1.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
