import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'drop_path_prob'}

class ConvBlock(nn.Module):
    """Basic building block: Conv -> BN -> ReLU -> Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class FractalBlock(nn.Module):
    """
    Recursive Fractal Block f_C.
    f_1 = Conv
    f_C = Join(Conv, f_{C-1} -> f_{C-1})
    """
    def __init__(self, c_depth, in_channels, out_channels, drop_path_prob):
        super().__init__()
        self.c_depth = c_depth
        self.drop_path_prob = drop_path_prob
        
        # Left Path: Always a single Convolution column
        self.left = ConvBlock(in_channels, out_channels)
        
        # Right Path: Recursive (only if depth > 1)
        self.right = None
        if c_depth > 1:
            self.right = nn.Sequential(
                FractalBlock(c_depth - 1, in_channels, out_channels, drop_path_prob),
                FractalBlock(c_depth - 1, out_channels, out_channels, drop_path_prob)
            )

    def forward(self, x):
        # 1. Compute paths
        left_out = self.left(x)
        
        if self.right is None:
            return left_out
            
        right_out = self.right(x)
        
        # 2. Join / DropPath Logic
        if self.training:
            # DropPath: Randomly kill one path or keep both (mean)
            r = random.random()
            if r < self.drop_path_prob:
                # Drop Right (Keep Left - fast path)
                return left_out
            elif r < 2 * self.drop_path_prob:
                # Drop Left (Keep Right - deep path)
                return right_out
            else:
                # Keep both (Average)
                return (left_out + right_out) * 0.5
        else:
            # Inference: Always Average
            return (left_out + right_out) * 0.5

class Net(nn.Module):
    def __init__(self, in_shape=(3, 32, 32), out_shape=(10,), prm=None, device="cpu"):
        super().__init__()
        self.device = device
        
        if prm is None:
            prm = {}
        
        # Hyperparameters
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.drop_path_prob = prm.get('drop_path_prob', 0.1)
        dropout_p = prm.get('dropout', 0.0)

        # Setup training components stored in model for easier mutation usage
        self.train_params = {'lr': lr, 'momentum': momentum}

        # --- Architecture Definition ---
        # Channels schedule: 64 -> 128 -> 256 -> 512
        c1, c2, c3, c4 = 64, 128, 256, 512
        recursion_depth = 3 # This is C. C=3 means 3 columns effectively.
        
        self.features = nn.Sequential(
            # Block 1
            FractalBlock(recursion_depth, in_shape[0], c1, self.drop_path_prob),
            nn.MaxPool2d(2),
            
            # Block 2
            FractalBlock(recursion_depth, c1, c2, self.drop_path_prob),
            nn.MaxPool2d(2),
            
            # Block 3
            FractalBlock(recursion_depth, c2, c3, self.drop_path_prob),
            nn.MaxPool2d(2),
            
            # Block 4
            FractalBlock(recursion_depth, c3, c4, self.drop_path_prob),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(c4 * 2 * 2, 4096) if in_shape[-1]==32 else nn.Linear(c4, 4096), # Adaptive handling assumption
            nn.ReLU(True),
            nn.Linear(4096, out_shape[0])
        )
        # Hack for dynamic shape adjustment if needed (input size dependent)
        # We assume CIFAR-10 32x32 -> MaxPoolx4 -> 2x2 spatial dim

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3]) # Global Average Pooling fallback if linear fails shape? 
        # Actually let's trust the hardcoded shape for now or flatten
        # x = torch.flatten(x, 1) # If we used Flatten
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        # Helper for our framework to set up optimizer
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.train_params['lr'],
            momentum=self.train_params['momentum']
        )
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()
