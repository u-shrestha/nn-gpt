import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=17, translate=(0.06, 0.14), scale=(0.87, 1.3), shear=(4.44, 6.99)),
    transforms.RandomRotation(degrees=12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
