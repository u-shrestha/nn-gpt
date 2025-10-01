import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.84),
    transforms.RandomAffine(degrees=18, translate=(0.18, 0.14), scale=(0.93, 1.38), shear=(2.41, 5.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
