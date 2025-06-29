import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), RandomHorizontalFlip(p=0.5), RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)]),
    transforms.RandomCrop(size=(46, 21)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
