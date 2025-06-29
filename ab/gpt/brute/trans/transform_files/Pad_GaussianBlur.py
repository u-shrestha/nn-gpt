import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=3, fill=(69, 49, 177), padding_mode=symmetric),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.4, 0.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
