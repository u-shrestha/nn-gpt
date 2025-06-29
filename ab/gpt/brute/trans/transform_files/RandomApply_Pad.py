import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomApply(transforms=[RandomHorizontalFlip(p=0.5)], p=0.67),
    transforms.Pad(padding=6, fill=(192, 96, 98), padding_mode=edge),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
