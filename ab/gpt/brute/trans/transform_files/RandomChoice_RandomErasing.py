import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomChoice(transforms=[RandomHorizontalFlip(p=0.5)]),
    transforms.RandomErasing(p=0.85, scale=(0.05, 0.1), ratio=(1.58, 1.51), value=random),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
