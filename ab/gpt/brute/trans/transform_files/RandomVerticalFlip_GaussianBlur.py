import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.48),
    transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 0.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
