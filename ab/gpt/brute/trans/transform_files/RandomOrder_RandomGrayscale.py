import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None), RandomHorizontalFlip(p=0.5)]),
    transforms.RandomGrayscale(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
