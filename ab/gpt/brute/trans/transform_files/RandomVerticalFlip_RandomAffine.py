import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.42),
    transforms.RandomAffine(degrees=5, translate=(0.11, 0.07), scale=(0.89, 0.83), shear=2.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
