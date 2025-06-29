import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=11),
    transforms.RandomVerticalFlip(p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
