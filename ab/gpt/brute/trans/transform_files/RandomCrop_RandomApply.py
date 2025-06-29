import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(51, 64)),
    transforms.RandomApply(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)], p=0.34),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
