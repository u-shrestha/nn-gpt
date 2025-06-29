import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[RandomHorizontalFlip(p=0.5), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), RandomHorizontalFlip(p=0.5)]),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.91), ratio=(0.95, 1.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
