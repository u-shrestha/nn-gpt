import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.53, p=0.36),
    transforms.RandomAffine(degrees=12, translate=(0.07, 0.07), scale=(1.07, 1.42), shear=(2.33, 6.21)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
