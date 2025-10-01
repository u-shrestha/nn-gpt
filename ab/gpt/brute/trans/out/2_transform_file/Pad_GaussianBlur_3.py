import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(79, 120, 39), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.29, 1.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
