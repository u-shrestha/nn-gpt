import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.Pad(padding=0, fill=(144, 27, 226), padding_mode=reflect),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
