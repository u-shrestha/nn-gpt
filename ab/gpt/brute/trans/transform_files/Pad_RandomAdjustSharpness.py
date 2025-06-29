import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=3, fill=(6, 117, 41), padding_mode=edge),
    transforms.RandomAdjustSharpness(sharpness_factor=1.16, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
