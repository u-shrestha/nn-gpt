import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.21, p=0.69),
    transforms.RandomAdjustSharpness(sharpness_factor=0.52, p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
