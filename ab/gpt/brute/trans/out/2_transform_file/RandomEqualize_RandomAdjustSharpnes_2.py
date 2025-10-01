import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.79),
    transforms.RandomAdjustSharpness(sharpness_factor=0.65, p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
