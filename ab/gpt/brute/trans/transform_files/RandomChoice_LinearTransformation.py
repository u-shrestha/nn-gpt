import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomChoice(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)]),
    transforms.LinearTransformation(transformation_matrix=tensor([[-162.8646,  -45.6611,  111.4824],
        [-145.2005,    4.4378,   74.5629],
        [-120.3888,   60.3512,  -60.7043]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
