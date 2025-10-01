import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.07, 0.02), scale=(1.2, 1.72), shear=(0.09, 6.71)),
    transforms.Pad(padding=4, fill=(207, 134, 228), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
