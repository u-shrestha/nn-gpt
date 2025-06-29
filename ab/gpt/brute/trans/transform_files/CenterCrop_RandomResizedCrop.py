import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=55),
    transforms.RandomResizedCrop(size=61, scale=(0.69, 0.94), ratio=(0.75, 1.05)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
