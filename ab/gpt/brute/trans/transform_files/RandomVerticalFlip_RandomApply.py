import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.63),
    transforms.RandomApply(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)], p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
