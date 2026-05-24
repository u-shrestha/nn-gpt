import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.79),
    transforms.RandomEqualize(p=0.26),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.67, 1.65)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
