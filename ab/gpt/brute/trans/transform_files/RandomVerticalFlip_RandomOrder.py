import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.18),
    transforms.RandomOrder(transforms=[RandomHorizontalFlip(p=0.5), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)]),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
