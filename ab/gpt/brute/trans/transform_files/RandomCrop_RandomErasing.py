import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(30, 63)),
    transforms.RandomErasing(p=0.64, scale=(0.06, 0.12), ratio=(0.34, 1.68), value=(0, 0, 0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
