import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.11, 0.12), scale=(1.14, 1.44), shear=(3.57, 9.54)),
    transforms.Pad(padding=5, fill=(139, 233, 127), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
