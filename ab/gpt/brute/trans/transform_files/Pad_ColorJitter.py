import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=8, fill=(9, 240, 69), padding_mode=constant),
    transforms.ColorJitter(brightness=0.91, contrast=1.17, saturation=0.95, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
