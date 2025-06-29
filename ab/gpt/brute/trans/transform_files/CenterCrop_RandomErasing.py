import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomErasing(p=0.66, scale=(0.09, 0.13), ratio=(3.15, 1.19), value=random),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
