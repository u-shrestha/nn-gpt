import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(157, 156, 78), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.93, 1.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
