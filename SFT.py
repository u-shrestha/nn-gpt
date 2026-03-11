import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import ab.nn.api as lemur
import pandas as pd
from datasets import Dataset
from ab.gpt.brute.fract.backbone.NNAlterBN import filter_backbones_by_size
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

import ast
import textwrap

# ================= Config=================
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
# MODEL_ID = "ABrain/NNGPT-UniqueArch-Rag"
OUTPUT_DIR = "./lora_test_results"
TRAIN_SAMPLES = 300
MAX_STEPS = 100
available_backbones = ['convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_t', 'swin_v2_t']

available_patterns = ['Parallel_Triple', 'Ensemble_Backbones_to_Fractal', 'Split_A_Parallel_BF']
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
    pass

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
        super().__init__()
        self.device = device
        self.use_amp = prm.get("use_amp", False)
        pass

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape)==4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
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
        pass

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

torchvision_skeleton_code = """
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
"""


# Input Context
# The following code skeleton provides fixed infrastructure (Wrappers, AMP, Training Loop). You must keep all existing code intact and ONLY implement the missing logic.

# [CODE SKELETON START]
# {skeleton_code}
# [CODE SKELETON END]
# TorchVision Wrapper:
# {torchvision_skeleton_code}
prompt_template = """
Role & Context
You are an expert AI architect specialized in deep learning. Your task is to implement the missing components (`pass` blocks) of a strict PyTorch model skeleton achieves an accuracy of {accuracy}.

Input Context
The following code skeleton provides fixed infrastructure (Wrappers, AMP, Training Loop). You must keep all existing code intact and ONLY implement the missing logic.

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

Implementation Requirements

1. **Implement `drop_conv3x3_block`**:
   - Return an `nn.Sequential` block containing a valid chain of Conv2d, BatchNorm, Activations (e.g., ReLU/SiLU), and Dropout.

2. **Implement `Net.__init__`**:
   - select `self.pattern from "{available_patterns}"` 
   - available_backbones is [{available_backbones}]
   - Initialize `self.backbone_a` and `self.backbone_b` from `available_backbones` using the `TorchVision` wrapper, Use specific model names and correct `in_channels` as the parameter.
   - Initialize `self.features` as an `nn.Sequential` of `FractalUnit` modules with growing channels, use only one or two layers of FractalUnit to avoid excessively small output.
   - Call `self.infer_dimensions_dynamically(in_shape, out_shape[0])` and initialize `self._scaler`.

3. **Implement `Net.forward` (CRITICAL)**:
   - Implement the forward pass based on the selected `self.pattern`
   - **Dimension Safety**: If the topology feeds the output of a Backbone/Vector into `self.features` (Fractal), you MUST reshape the vector to 4D (`unsqueeze`) and upscale it using `torch.nn.functional.interpolate` to at least `(14, 14)` before passing it to the fractal network.
   - Example safety check for Vector inputs:
     `if tensor.dim() == 2: tensor = tensor.unsqueeze(-1).unsqueeze(-1)`
     `if tensor.shape[-1] < 14: tensor = F.interpolate(tensor, size=(14,14))`

Output Instructions:
You must output ONLY the implementation of the three missing components, wrapped in the following XML tags:
1. <block> ... code for drop_conv3x3_block ... </block>
2. <init> ... code for Net.__init__ ... </init>
3. <forward> ... code for Net.forward ... </forward>

Do not output the full file. Do not output any markdown formatting (like ```python).
"""
#    - select `self.pattern from "{available_patterns}"` 
#    - Must strictly follow the topology from "{available_patterns}"


def parse_nn_code(code_str):
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        block_code = None
        init_code = None
        forward_code = None

        def get_source(node):
            if hasattr(node, 'end_lineno'):
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
            else:
                return "\n".join(lines[node.lineno - 1 :])

        for node in tree.body:
            # if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            #     print(node.name)
            if isinstance(node, ast.FunctionDef) and node.name == 'drop_conv3x3_block':
                block_code = get_source(node)

            elif isinstance(node, ast.ClassDef) and node.name == 'Net':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        if sub_node.name == '__init__':
                            init_code = get_source(sub_node)
                        elif sub_node.name == 'forward':
                            forward_code = get_source(sub_node)

        if block_code: block_code = textwrap.dedent(block_code).strip()
        if init_code: init_code = textwrap.dedent(init_code).strip()
        if forward_code: forward_code = textwrap.dedent(forward_code).strip()

        return block_code, init_code, forward_code

    except Exception as e:
        print(f"AST Parsing Failed: {e}")
        print(f"Code snippet: {code_str[:100]}...")
        return None, None, None


def build_dataset(tokenizer):
    print(f"extracting data from Lemur...")
    df = lemur.data(
        # only_best_accuracy=True,
        task='img-classification',
        dataset='cifar-10',
        metric='acc',
        # nn_prefixes=tuple("rl-back-init")
        nn_prefixes=("rl-bb-init",)
    )
    print(f"extracted {len(df)} samples.")
    print(df.iloc[0])

    df = df.sample(n=min(len(df), TRAIN_SAMPLES))

    # available_backbones = filter_backbones_by_size(max_params_millions=30)
    # print(f"Available Backbones : {available_backbones}")
    formatted_data = []
    for _, row in df.iterrows():
    # for _, row in df.iterrows():
        full_code = row['nn_code']
        accuracy = row['accuracy']
        # print(full_code)

        block_code, init_code, forward_code = parse_nn_code(full_code)
        assistant_response = f"<block>\n{block_code}\n</block>\n<init>\n{init_code}\n</init>\n<forward>\n{forward_code}\n</forward>"
        messages = [
            {"role": "user", "content": prompt_template.format(accuracy=accuracy, skeleton_code=skeleton_code, available_patterns=", ".join(available_patterns), available_backbones=", ".join(available_backbones))},
            {"role": "assistant", "content": assistant_response}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)

def formatting_prompts_func(example):
    output_texts = []
    for messages in example['messages']:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        output_texts.append(text)
    return output_texts

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    bnb_confi_1 = BitsAndBytesConfig(
        load_in_8bit=True,  
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_confi_1, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # collator = DataCollatorForLanguageModeling(
    #     pad_token_id=pad_token_id,
    #     completion_only_loss=True,
    #     pad_to_multiple_of=8,
    #     return_tensors="pt"
    # )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 4096
    training_args = SFTConfig(
        output_dir="./lora_test",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        # max_steps=100,
        logging_steps=5,
        bf16=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns = True,
        max_seq_length = 5000,
        dataset_text_field = "text",
        packing=False,
        gradient_checkpointing=True
    )
    # training_args = TrainingArguments(
    #     output_dir="./lora_test",
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=4,
    #     learning_rate=5e-5,
    #     num_train_epochs=5,
    #     # max_steps=100,
    #     logging_steps=5,
    #     bf16=True,
    #     save_strategy="no",
    #     report_to="none",
    #     remove_unused_columns = False,
    #     # max_seq_length = 4096,
    #     dataset_text_field = "text",
    # )
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    response_template = "\n## Response:" 

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        # peft_config=lora_config,
        args=training_args,
        data_collator=collator,
        # dataset_text_field="text",
        # completion_only_loss=True,
    )

    # print("\n" + "="*30 + " format review " + "="*30)
    # sample_preview = dataset[:1]

    # from pprint import pprint
    # pprint(sample_preview)
    # print(sample_preview)
    print("="*78 + "\n")

    messages = [
        {
            "role": "user",
             "content": prompt_template.format(accuracy=0.90, skeleton_code=skeleton_code, available_patterns=", ".join(available_patterns), available_backbones=", ".join(available_backbones)),
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    input_length = inputs.input_ids.shape[1]

    print("Testing ability before training...................\n")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
            # **inputs,
            # max_new_tokens=2048,
            # do_sample=True,
            # temperature=0.2,
            # top_p=0.9,
            # top_k=50,
            # eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_ids = output_ids[0][input_length:]
    # generated_ids = output_ids[0]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Output:")
    print(response)
    # response_split = full_response.split("<｜Assistant｜>")
    # print(response_split[1])
    print("="*30)

    print("Tring...")
    model.train()
    trainer.train()
    print("Completed Training.")

    print("Generating final code...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
            # **inputs,
            # max_new_tokens=2048,
            # do_sample=True,
            # temperature=0.2,
            # top_p=0.9,
            # top_k=50,
            # eos_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = output_ids[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("Reslut:")
    print(response)
    # response_split = full_response.split("<｜Assistant｜>")
    # final_code = response_split[-1] if len(response_split) > 1 else full_response

if __name__ == "__main__":
    main()