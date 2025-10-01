import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=24, translate=(0.04, 0.01), scale=(0.85, 1.41), shear=(2.46, 5.57)),
    transforms.Pad(padding=0, fill=(198, 166, 24), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
