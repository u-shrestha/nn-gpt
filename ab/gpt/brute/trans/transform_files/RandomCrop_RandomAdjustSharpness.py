import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(39, 37)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
