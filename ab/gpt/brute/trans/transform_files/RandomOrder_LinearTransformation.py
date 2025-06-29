import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomOrder(transforms=[RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0), RandomHorizontalFlip(p=0.5), RandomHorizontalFlip(p=0.5)]),
    transforms.LinearTransformation(transformation_matrix=tensor([[  78.6453,   26.5672,  -19.0863],
        [ 154.0002, -130.8779,   82.4852],
        [ -29.9811,  124.1538, -119.2931]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
