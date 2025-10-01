import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.04, 0.18), scale=(1.19, 1.27), shear=(3.88, 5.5)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
