import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.14, saturation=1.14, hue=0.07),
    transforms.Pad(padding=5, fill=(131, 181, 230), padding_mode='symmetric'),
    transforms.RandomCrop(size=32),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
