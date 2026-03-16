import ast
import textwrap

available_backbones = ['convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_t', 'swin_v2_t']

available_patterns = [
    'Parallel_Triple', 
    'Backbone_A_to_Fractal', 
    'Backbone_B_to_Fractal', 
    'Dual_Backbone_Fuse_Then_Fractal',
    'Fractal_Then_Dual_Backbone',
    'Split_Stem_Parallel_Fuse'
]

skeleton_code = """import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# ==========================================
# 1. FIXED INFRASTRUCTURE (DO NOT MODIFY)
# ==========================================
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4: return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3: return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# ==========================================
# 2. DYNAMIC COMPONENTS (TO BE IMPLEMENTED)
# ==========================================

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        

    def infer_dimensions_dynamically(self, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            c, h, w = self._input_spec
            dummy = torch.zeros(1, c, h, w).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss): continue
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'): train_iter.shutdown()
            del train_iter
            gc.collect()
"""

prompt_template="""
### Role & Context
You are a Senior AI Architect. Your task is to implement a **specific** model instance based on a strict skeleton to achieve an accuracy of {accuracy}. 

### Task Overview
Complete the three missing components. **DO NOT** write generic code. You must implement the architecture using the target design pattern provided.

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Technical Specifications (MANDATORY REQUIREMENTS)

1. **Target Pattern: `{target_pattern}`**
   - YOU MUST explicitly set `self.pattern = '{target_pattern}'` inside `__init__`.
   - YOU MUST implement the logic for this specific pattern throughout the code.
   - **CRITICAL REQUIREMENT**: DO NOT just blindly copy the standard Parallel_Triple structure. You MUST be highly creative and design a truly unique structural flow in `forward`. Vary your module usage and connection topology!

2. **Component: `drop_conv3x3_block`**
   - Implement starting with `def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):`.
   - Return an `nn.Sequential` block (Conv2d -> BatchNorm2d -> Activation -> Dropout2d).

3. **Component: `Net.__init__`**
   - Implement starting with `def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:`.
   - **MANDATORY**: `self.pattern = '{target_pattern}'`
   - **Backbone Selection**: Choose EXACTLY TWO models from [{available_backbones}].
   - **Initialization**: 
     - Initialize `self.backbone_a` and `self.backbone_b` using `TorchVision(model='...', in_channels=...)`.
     - Initialize `self.features` (1-2 `FractalUnit` layers).
     - Call `self.infer_dimensions_dynamically(in_shape, out_shape[0])`.
   - **Example Implementation Fragment**:
     ```python
     self.pattern = '{target_pattern}'
     self.backbone_a = TorchVision(model='resnet18', in_channels=in_shape[1]).to(device)
     ...
     ```

4. **Component: `Net.forward`**
   - Implement starting with `def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:`.
   - **Flow Control**: Implement the data flow for `{target_pattern}`. Use `adaptive_pool_flatten` for module outputs before fusion.
   - **Fusion Patterns Logic Blueprint**:
     * `Parallel_Triple`: `Result = Concat(backbone_a(x), backbone_b(x), features(x))`
     * `Backbone_A_to_Fractal`: `Result = features(backbone_a(x))` (Sequential flow)
     * `Split_Stem_Parallel_Fuse`: `stem_out = STEM(x); Result = Concat(backbone_a(stem_out), backbone_b(stem_out))`
   - **CRITICAL - NO GHOSTING**: You MUST use ALL defined components in the `forward` pass.
   - **CRITICAL RESTRICTION**: You MUST build the computational graph directly without using ANY `if self.pattern == ...` control flow or dynamic loops (like `getattr`/`hasattr`) inside `forward`.
   - **PARAM REMINDER**: Always pass `in_channels=...` when creating `TorchVision` models.

### Output Requirement (STRICT)
Output ONLY the implementation within the XML tags. Each tag MUST contain the complete function/method definition (signature and body). No markdown, no conversation.

<block>
# Full drop_conv3x3_block implementation
</block>
<init>
# Full __init__ implementation
</init>
<forward>
# Full forward implementation
</forward>
"""

def parse_nn_code(code_str):
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        block_code = None
        init_code = None
        forward_code = None

        def get_source(node):
            return ast.get_source_segment(code_str, node)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'drop_conv3x3_block':
                block_code = get_source(node)

            elif isinstance(node, ast.ClassDef) and node.name == 'Net':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        if sub_node.name == '__init__':
                            init_code = get_source(sub_node)
                        elif sub_node.name == 'forward':
                            forward_code = get_source(sub_node)

        def clean_code(c):
            return textwrap.dedent(c).strip() if c else None

        return clean_code(block_code), clean_code(init_code), clean_code(forward_code)

    except Exception as e:
        print(f"AST Parsing Failed: {e}")
        print(f"Code snippet: {code_str[:100]}...")
        return None, None, None
