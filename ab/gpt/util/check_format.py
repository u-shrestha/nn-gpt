import sys
from pathlib import Path
import os

# Dynamically add workspace roots to path based on script location
script_path = Path(__file__).resolve()
gw_root = script_path.parent.parent.parent.parent  # Points to .../nn-gpt
ds_root = gw_root.parent / 'nn-dataset'          # Points to .../nn-dataset

sys.path.append(str(ds_root))
sys.path.append(str(gw_root))

# Also add the B0 output directory dynamically for new_nn import
# Path: .../nn-gpt/out/nngpt/llm/epoch/A0/synth_nn/B0
b0_path = gw_root / 'out' / 'nngpt' / 'llm' / 'epoch' / 'A0' / 'synth_nn' / 'B0'
sys.path.append(str(b0_path))

try:
    import new_nn
    model = new_nn.load(device='cpu')
    print("Model loaded successfully.")
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            print(f"Parameter: {name}")
            print(f"  Layout: {param.layout}")
            print(f"  Is Sparse: {param.is_sparse}")
            print(f"  Format: {param.layout}")
            # Check if it has zeros
            sparsity = (param == 0).float().mean().item()
            print(f"  Actual Zero Sparsity: {sparsity:.2%}")
            break
except Exception as e:
    print(f"Error: {e}")
