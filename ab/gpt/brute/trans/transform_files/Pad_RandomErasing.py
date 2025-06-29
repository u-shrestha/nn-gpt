import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=4, fill=(206, 49, 183), padding_mode=constant),
    transforms.RandomErasing(p=0.22, scale=(0.32, 0.26), ratio=(1.71, 2.14), value=(0, 0, 0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
