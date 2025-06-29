import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=1, fill=(140, 193, 172), padding_mode=constant),
    transforms.RandomChoice(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)]),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
