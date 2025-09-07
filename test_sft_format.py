#!/usr/bin/env python3
"""
修正SFT数据格式问题 - 专门针对SFTTrainer
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from pathlib import Path
import sys
import os
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

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

def create_sft_format_dataset():
    """创建正确格式的SFT数据集"""
    
    print("=== 创建SFT格式数据集 ===")
    
    # 初始化tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置路径
    train_config_path = conf_test_dir / 'NN_gen.json'
    
    # 使用原始的数据处理器
    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=10)  # 只用10个样本测试
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)
    
    print(f"原始数据集大小: {len(dataset)}")
    
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
        "Answer:"
    ]
    
    def build_sft_dataset(example):
        code = example["response"]
        try:
            parsed = json.loads(code)
            code = parsed["nn_code"]
            masked_model, masked_code = mask_layer_definitions(code)
            
            prompt = "\n".join([
                line if "{masked_prompt}" not in line else line.format(masked_prompt=masked_model)
                for line in PROMPT_TEMPLATE
            ])
            
            # SFTTrainer期望的格式：完整的对话文本
            full_text = f"{prompt}\n{masked_code}"
            
            return {
                "text": full_text  # SFTTrainer期望的字段名
            }
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return {
                "text": ""
            }
    
    # 应用转换
    sft_dataset = dataset.map(build_sft_dataset, remove_columns=dataset.column_names)
    sft_dataset = sft_dataset.filter(lambda x: x["text"] != "")
    
    print(f"SFT格式数据集大小: {len(sft_dataset)}")
    
    if len(sft_dataset) > 0:
        sample = sft_dataset[0]
        print(f"\n=== 第一个SFT样本（长度：{len(sample['text'])}）===")
        print(sample["text"][:500] + "...")
    
    return sft_dataset

def test_sft_training():
    """测试SFT训练"""
    
    print("\n=== 测试SFT训练 ===")
    
    # 创建数据集
    dataset = create_sft_format_dataset()
    
    if len(dataset) == 0:
        print("❌ 数据集为空，无法训练")
        return
    
    # 初始化模型和tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # 使用float32避免MPS问题
        device_map={"": "mps"},
        trust_remote_code=True
    )
    
    # LoRA配置
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="./sft_test_outputs",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=999,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=False,
        report_to=[]
    )
    
    # SFT训练器 - 使用最简单的参数
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config
    )
    
    print("开始SFT训练...")
    result = trainer.train()
    print(f"训练完成！最终loss: {result.training_loss}")
    
    # 测试训练后的模型
    print("\n=== 测试训练后的模型 ===")
    
    test_prompt = """Complete the missing layers in this PyTorch neural network:

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        [MASKED LAYER]
        [MASKED LAYER]
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

Answer:"""
    
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("生成结果:")
    print(generated_text)

if __name__ == "__main__":
    test_sft_training()
