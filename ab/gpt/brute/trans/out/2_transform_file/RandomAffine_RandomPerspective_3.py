import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=17, translate=(0.2, 0.11), scale=(1.11, 1.74), shear=(0.46, 8.61)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
