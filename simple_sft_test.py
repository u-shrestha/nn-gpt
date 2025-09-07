#!/usr/bin/env python3

"""
简化版SFT训练 - 只用很少的数据进行测试
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import json

def simple_sft_test():
    print("=== 简化版SFT测试 ===")
    
    # 模型和tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 非常简单的训练数据
    simple_data = [
        {
            "prompt": "写一个PyTorch线性层，输入维度10，输出维度5：",
            "response": """```python
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)
```"""
        },
        {
            "prompt": "创建一个PyTorch卷积层，输入通道3，输出通道64：",
            "response": """```python
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)
```"""
        }
    ]
    
    # 创建数据集
    dataset = Dataset.from_list(simple_data)
    
    def format_data(example):
        # SFTTrainer期待的格式
        return {
            "prompt": example["prompt"],
            "completion": example["response"]
        }
    
    dataset = dataset.map(format_data)
    
    # 非常保守的LoRA设置
    peft_config = LoraConfig(
        r=4,  # 很小的秩
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # 只训练少数模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    # 非常保守的训练设置
    training_args = TrainingArguments(
        output_dir="./simple_sft",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # 很小的学习率
        logging_steps=1,
        save_steps=999,  # 不保存中间checkpoint
        remove_unused_columns=False,
        fp16=False,
        report_to=[]  # 禁用wandb
    )
    
    # SFT训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config
    )
    
    print("开始简化SFT训练...")
    trainer.train()
    
    print("\n测试训练后的效果:")
    test_prompt = "写一个PyTorch ReLU激活层："
    
    messages = [{"role": "user", "content": test_prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("生成结果:")
    print(generated)

if __name__ == "__main__":
    simple_sft_test()
