import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
