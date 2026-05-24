import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.37),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.42, 1.88)),
    transforms.RandomVerticalFlip(p=0.39),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
