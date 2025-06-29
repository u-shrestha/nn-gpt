import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomVerticalFlip(p=0.4),
    transforms.LinearTransformation(transformation_matrix=tensor([[-205.5293,   66.0382, -132.8017],
        [  19.3064, -158.8101,  -64.9626],
        [  54.9267, -101.1798,   52.8536]])),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
