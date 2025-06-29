import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=8, fill=(129, 106, 123), padding_mode=constant),
    transforms.RandomGrayscale(p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
