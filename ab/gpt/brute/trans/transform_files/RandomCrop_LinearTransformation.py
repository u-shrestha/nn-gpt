import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomCrop(size=(59, 60)),
    transforms.LinearTransformation(transformation_matrix=tensor([[ 145.2546,   57.2233,  121.9792],
        [ -44.1998,  142.2833, -110.5613],
        [ -31.8771,   52.2048,   -4.6164]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
