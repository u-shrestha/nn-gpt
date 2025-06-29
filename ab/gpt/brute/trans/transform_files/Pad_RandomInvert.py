import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=3, fill=(68, 63, 81), padding_mode=constant),
    transforms.RandomInvert(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
