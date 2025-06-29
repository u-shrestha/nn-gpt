import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.78),
    transforms.RandomErasing(p=0.38, scale=(0.26, 0.3), ratio=(2.72, 1.84), value=(0, 0, 0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
