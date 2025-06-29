import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=9, fill=(209, 64, 14), padding_mode=reflect),
    transforms.RandomApply(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)], p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
