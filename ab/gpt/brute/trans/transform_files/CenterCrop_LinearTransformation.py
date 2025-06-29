import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.CenterCrop(size=13),
    transforms.LinearTransformation(transformation_matrix=tensor([[-162.2758,   15.7548,  -65.0377],
        [-120.0712,   80.4593,   95.1069],
        [-198.0902,   52.7685,    4.2574]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
