import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Callable

class FractalDropPath(nn.Module):
    """
    DropPath Regularization Layer.
    Randomly drops inputs during training to prevent co-adaptation.
    """
    def __init__(self, drop_prob: float = 0.15):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Inference: Average all paths
        if not self.training:
            return torch.stack(inputs).mean(dim=0)
            
        # Training: Randomly drop paths
        # We must ensure at least one path survives.
        n = len(inputs)
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        
        if mask.sum() == 0:
             # Force keep one random path
             idx = torch.randint(0, n, (1,)).item()
             mask[idx] = 1.0
             
        active_inputs = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active_inputs).mean(dim=0)

class FractalBlock(nn.Module):
    """
    A recursive Fractal Block f_C.
    - C=1: Base Module (typically Conv-BN-ReLU)
    - C>1: Join( f_{C-1}, f_{C-1} o f_{C-1} )
    """
    def __init__(self, n_columns: int, base_module_fn: Callable[[], nn.Module], dropout_prob: float = 0.1):
        super().__init__()
        self.n_columns = n_columns
        
        if n_columns == 1:
            self.net = base_module_fn()
        else:
            # Recursive structure
            # Left Path: Shallower (f_{C-1})
            self.left = FractalBlock(n_columns - 1, base_module_fn, dropout_prob)
            
            # Right Path: Deeper (f_{C-1} stacked)
            self.right_1 = FractalBlock(n_columns - 1, base_module_fn, dropout_prob)
            self.right_2 = FractalBlock(n_columns - 1, base_module_fn, dropout_prob)
            
            self.join = FractalDropPath(drop_prob=dropout_prob)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_columns == 1:
            return self.net(x)
            
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        return self.join([out_left, out_right])
        
    def forward_shallowest(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes ONLY the shallowest path (Column 1) for rapid evaluation.
        """
        if self.n_columns == 1:
            return self.net(x)
        return self.left.forward_shallowest(x)

def simplify_name(clean_code: str) -> str:
    """Helper for the LLM to know what class to look for."""
    return "FractalBlock"
