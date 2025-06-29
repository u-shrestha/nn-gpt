import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=62),
    transforms.RandomAffine(degrees=17, translate=(0.17, 0.11), scale=(1.12, 1.01), shear=3.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
