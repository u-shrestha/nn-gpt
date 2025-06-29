import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=39),
    transforms.RandomCrop(size=(39, 26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
