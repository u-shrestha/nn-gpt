import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=4, fill=(165, 143, 113), padding_mode=constant),
    transforms.RandomResizedCrop(size=50, scale=(0.57, 0.94), ratio=(0.99, 1.08)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
