#!/usr/bin/env python3
"""
调试数据集格式 - 检查SFT训练数据的问题
"""

import json
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.Util import extract_code
from ab.gpt.util.Const import conf_test_dir

def mask_layer_definitions(code):
    """mask layer definitions and return original layers"""
    lines = code.split('\n')
    masked_lines = []
    original_layers = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('self.') and ('=' in stripped):
            # Extract the layer definition
            original_layers.append(stripped)
            # Replace with mask
            masked_lines.append(line.replace(stripped, '[MASKED LAYER]'))
        else:
            masked_lines.append(line)
    
    masked_code = '\n'.join(masked_lines)
    original_code = '\n'.join(original_layers)
    
    return masked_code, original_code

def format_for_rl(example):
    """格式化数据用于RL训练"""
    prompt = example["prompt"] if example["prompt"] else ""
    response = example["response"] if example["response"] else ""
    return {
        "prompt": prompt,
        "response": response.strip(),
        "completion": response.strip(),
    }

def debug_dataset():
    """调试数据集格式"""
    
    print("=== 调试数据集格式 ===")
    
    # 初始化tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置路径
    train_config_path = conf_test_dir / 'NN_gen.json'
    
    print(f"配置文件路径: {train_config_path}")
    print(f"配置文件存在: {train_config_path.exists()}")
    
    if not train_config_path.exists():
        print("❌ 配置文件不存在！")
        return
    
    # 使用原始的数据处理器
    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=100)  # 只取100个样本调试
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)
    
    print(f"原始数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ 数据集为空！")
        return
    
    # 检查第一个样本
    first_sample = dataset[0]
    print("\n=== 第一个原始样本 ===")
    print(f"instruction: {first_sample.get('instruction', 'N/A')}")
    print(f"response: {first_sample.get('response', 'N/A')[:200]}...")
    print(f"text: {first_sample.get('text', 'N/A')[:200]}...")
    
    # 检查response的格式
    response = first_sample.get('response', '')
    if response:
        try:
            parsed = json.loads(response)
            print(f"\n解析后的response包含字段: {list(parsed.keys())}")
            if 'nn_code' in parsed:
                nn_code = parsed['nn_code']
                print(f"nn_code长度: {len(nn_code)}")
                print(f"nn_code前200字符:\n{nn_code[:200]}")
            else:
                print("❌ response中没有nn_code字段！")
        except json.JSONDecodeError as e:
            print(f"❌ 无法解析response为JSON: {e}")
            print(f"response内容: {response[:200]}")
    
    # 应用masking到少量样本
    dataset_small = dataset.select(range(5))
    
    # Prompt模板
    PROMPT_TEMPLATE = [
        "Complete the missing layers in this PyTorch neural network:",
        "",
        "{masked_prompt}",
        "",
        "Requirements:",
        "1. Must be valid Python/PyTorch code", 
        "2. Include necessary imports (torch, torch.nn)",
        "3. Define a class inheriting from nn.Module",
        "4. Implement __init__ and forward methods",
        "5. Replace each [MASKED LAYER] with the appropriate PyTorch layer definition",
        "",
        "Output as a complete code block between ```python and ```"
    ]
    
    def build_masked_dataset(example):
        code = example["response"]
        try:
            parsed = json.loads(code)
            code = parsed["nn_code"]
            masked_model, masked_code = mask_layer_definitions(code)
            return {
                "prompt": "\n".join([
                    line if "{masked_prompt}" not in line else line.format(masked_prompt=masked_model)
                    for line in PROMPT_TEMPLATE
                ]),
                "response": masked_code.strip()
            }
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return {
                "prompt": "",
                "response": ""
            }
    
    # 应用masking
    masked_dataset = dataset_small.map(build_masked_dataset, remove_columns=dataset_small.column_names)
    masked_dataset = masked_dataset.filter(lambda x: x["response"] != "")
    
    print(f"\n=== Masking后数据集大小: {len(masked_dataset)} ===")
    
    if len(masked_dataset) > 0:
        masked_sample = masked_dataset[0]
        print("\n=== 第一个masked样本 ===")
        print("Prompt:")
        print(masked_sample["prompt"])
        print("\nResponse:")
        print(masked_sample["response"])
        
        # 检查tokenization
        print("\n=== Tokenization测试 ===")
        prompt_tokens = tokenizer(masked_sample["prompt"], return_tensors="pt")
        response_tokens = tokenizer(masked_sample["response"], return_tensors="pt")
        
        print(f"Prompt token数量: {len(prompt_tokens.input_ids[0])}")
        print(f"Response token数量: {len(response_tokens.input_ids[0])}")
        
        # 检查SFTTrainer期望的格式
        print("\n=== SFTTrainer格式检查 ===")
        # SFTTrainer通常期望一个文本字段包含完整的对话
        formatted_text = masked_sample["prompt"] + masked_sample["response"]
        print(f"格式化文本长度: {len(formatted_text)}")
        
        # 格式化为RL格式
        rl_format = format_for_rl(masked_sample)
        print("\n=== RL格式 ===")
        for key, value in rl_format.items():
            print(f"{key}: {str(value)[:100]}...")

if __name__ == "__main__":
    debug_dataset()
