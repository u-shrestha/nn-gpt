import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomHorizontalFlip(p=0.38),
    transforms.LinearTransformation(transformation_matrix=tensor([[ 124.2709,  -70.1033,   29.9096],
        [   4.2203,  -60.3325,    7.8376],
        [-198.3988,   14.2517,   56.7522]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
