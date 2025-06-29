import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomApply(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)], p=0.8),
    transforms.LinearTransformation(transformation_matrix=tensor([[-137.0936, -136.1961,  -48.8208],
        [ 208.9477,   59.0267,   37.8185],
        [ 158.7899,   61.1970,   -8.8058]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
