import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.RandomAffine(degrees=8, translate=(0.19, 0.04), scale=(1.08, 0.86), shear=-5.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
