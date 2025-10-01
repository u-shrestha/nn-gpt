import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(58, 85, 165), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.2), scale=(1.2, 1.3), shear=(2.08, 9.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
