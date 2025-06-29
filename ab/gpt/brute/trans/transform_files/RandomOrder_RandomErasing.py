import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)]),
    transforms.RandomErasing(p=0.35, scale=(0.11, 0.04), ratio=(0.37, 2.62), value=random),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
