import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=64),
    transforms.RandomSolarize(threshold=30, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
