import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=9, fill=(225, 7, 51), padding_mode=edge),
    transforms.RandomAffine(degrees=8, translate=(0.13, 0.05), scale=(0.97, 0.86), shear=-9.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
