import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(125, 117, 195), padding_mode='constant'),
    transforms.RandomEqualize(p=0.35),
    transforms.RandomSolarize(threshold=96, p=0.22),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
