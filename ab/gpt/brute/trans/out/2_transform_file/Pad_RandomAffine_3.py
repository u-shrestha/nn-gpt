import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(2, 133, 198), padding_mode='constant'),
    transforms.RandomAffine(degrees=25, translate=(0.15, 0.08), scale=(1.02, 1.53), shear=(4.43, 7.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
