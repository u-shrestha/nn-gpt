import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.26, p=0.11),
    transforms.RandomAffine(degrees=15, translate=(0.02, 0.19), scale=(1.19, 1.38), shear=(4.28, 8.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
