import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.64),
    transforms.RandomEqualize(p=0.51),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
