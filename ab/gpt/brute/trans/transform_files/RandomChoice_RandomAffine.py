import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomChoice(transforms=[RandomHorizontalFlip(p=0.5)]),
    transforms.RandomAffine(degrees=12, translate=(0.05, 0.17), scale=(1.11, 1.19), shear=-0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
