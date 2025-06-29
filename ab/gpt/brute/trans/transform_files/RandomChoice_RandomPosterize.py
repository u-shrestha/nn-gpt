import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomChoice(transforms=[RandomHorizontalFlip(p=0.5)]),
    transforms.RandomPosterize(bits=7, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
