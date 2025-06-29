import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomChoice(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
