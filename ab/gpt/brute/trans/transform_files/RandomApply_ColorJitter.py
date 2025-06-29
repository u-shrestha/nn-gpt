import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomApply(transforms=[RandomHorizontalFlip(p=0.5)], p=0.27),
    transforms.ColorJitter(brightness=1.19, contrast=0.82, saturation=0.91, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
