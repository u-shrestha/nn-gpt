import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.17, p=0.38),
    transforms.RandomPosterize(bits=6, p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
