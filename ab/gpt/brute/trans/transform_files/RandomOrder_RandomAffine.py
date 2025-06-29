import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), RandomHorizontalFlip(p=0.5)]),
    transforms.RandomAffine(degrees=4, translate=(0.2, 0.14), scale=(0.98, 1.08), shear=2.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
