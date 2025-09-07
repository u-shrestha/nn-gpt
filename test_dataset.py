#!/usr/bin/env python3
"""
测试数据集加载和格式化
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.Code import extract_code
from ab.gpt.util.NNLayer import mask_layer_definitions
import json
from datasets import Dataset

# 设置路径
conf_test_dir = project_root / 'ab' / 'gpt' / 'conf' / 'prompt' / 'test'

def format_for_rl(example):
    """格式化数据用于RL训练"""
    prompt = example["prompt"] if example["prompt"] else ""
    response = example["response"] if example["response"] else ""
    return {
        "prompt": prompt,
        "response": response.strip(),
        "completion": response.strip(),
    }

def load_original_dataset(tokenizer):
    """使用原始TuneRL.py的数据获取代码"""
    
    # 配置路径
    train_config_path = conf_test_dir / 'NN_gen.json'
    
    print("正在获取原始数据集...")
    
    # 使用原始的数据处理器
    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=100000)
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)
    
    print(f"原始数据集大小: {len(dataset)}")
    
    # 大幅减少训练数据量，基于成功的简化SFT测试
    dataset = dataset.select(range(5)).shuffle()  # 只使用5个样本进行测试
    
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
            print(f"Error processing example: {e}")
            return {
                "prompt": "",
                "response": ""
            }
    
    # 应用masking
    masked_dataset = dataset.map(build_masked_dataset, remove_columns=dataset.column_names)
    dataset = masked_dataset.filter(lambda x: x["response"] != "")
    
    # 格式化为RL格式
    dataset = dataset.map(format_for_rl, remove_columns=dataset.column_names)
    
    print(f"处理后数据集大小: {len(dataset)}")
    
    return dataset

def main():
    print("=== 测试数据集加载 ===")
    
    # 加载模型和tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_original_dataset(tokenizer)
    
    # 显示前3个样本
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"\n=== 样本 {i+1} ===")
        print("Prompt:")
        print(example["prompt"][:500] + "..." if len(example["prompt"]) > 500 else example["prompt"])
        print("\nResponse:")
        print(example["response"][:300] + "..." if len(example["response"]) > 300 else example["response"])
        print("-" * 80)
    
    print(f"\n数据集准备完成！总共 {len(dataset)} 个样本")

if __name__ == "__main__":
    main()
