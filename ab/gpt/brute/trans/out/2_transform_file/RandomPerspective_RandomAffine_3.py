import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.28, p=0.79),
    transforms.RandomAffine(degrees=19, translate=(0.01, 0.08), scale=(1.03, 1.54), shear=(4.03, 5.36)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
