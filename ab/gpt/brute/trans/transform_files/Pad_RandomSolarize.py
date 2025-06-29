import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=4, fill=(150, 208, 169), padding_mode=edge),
    transforms.RandomSolarize(threshold=33, p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
