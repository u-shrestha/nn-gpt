import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.RandomOrder(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)]),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
