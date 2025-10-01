import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(74, 34, 62), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.18, 1.15)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
