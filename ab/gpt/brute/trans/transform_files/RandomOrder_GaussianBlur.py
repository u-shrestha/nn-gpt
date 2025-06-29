import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), RandomHorizontalFlip(p=0.5), RandomHorizontalFlip(p=0.5)]),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.67, 0.7)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
