import os

# Definition of the Initial Fractal Seed
fractal_seed_code = """
import torch
import torch.nn as nn
from modular.fractalnet import FractalBlock

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Helper to create base modules (Atom of the fractal)
        def conv_block_fn_1():
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )

        def conv_block_fn_2():
            return nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
        # Fractal Architecture
        # Block 1: 32 channels, 2 columns
        self.block1 = FractalBlock(n_columns=2, base_module_fn=conv_block_fn_1, dropout_prob=0.1)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 64 channels, 2 columns
        self.block2 = FractalBlock(n_columns=2, base_module_fn=conv_block_fn_2, dropout_prob=0.1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def forward_shallowest(self, x):
        # Proxy Evaluation Method
        x = self.block1.forward_shallowest(x)
        x = self.pool1(x)
        
        x = self.block2.forward_shallowest(x)
        x = self.pool2(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
"""

output_path = 'ab/gpt/brute/ga/modular/fractal_seed.py'

def generate_seed():
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(fractal_seed_code)
    print(f"Generated Fractal Seed at: {output_path}")

if __name__ == "__main__":
    generate_seed()
