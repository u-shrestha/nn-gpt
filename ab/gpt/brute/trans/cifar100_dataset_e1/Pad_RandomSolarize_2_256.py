import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(226, 161, 240), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=56, p=0.34),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
