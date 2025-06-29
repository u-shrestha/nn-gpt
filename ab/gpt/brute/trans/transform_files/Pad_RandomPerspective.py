import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=3, fill=(33, 39, 70), padding_mode=symmetric),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.75),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
