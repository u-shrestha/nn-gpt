import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.Pad(padding=1, fill=(1, 95, 44), padding_mode=reflect),
    transforms.LinearTransformation(transformation_matrix=tensor([[-146.9108, -141.1422,   89.5405],
        [-144.9285,  -19.2587,  100.2305],
        [ -35.9460,  144.2064,   -1.7033]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
