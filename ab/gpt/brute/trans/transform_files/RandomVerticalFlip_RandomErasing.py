import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.64),
    transforms.RandomErasing(p=0.73, scale=(0.16, 0.24), ratio=(1.9, 3.0), value=random),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
