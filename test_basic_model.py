#!/usr/bin/env python3

"""
测试基础模型的代码生成能力
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_basic_generation():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "auto"
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试prompt
    prompt = """请用PyTorch编写一个简单的卷积层：

要求：
1. 使用nn.Conv2d
2. 输入通道16，输出通道32
3. 卷积核大小3x3
4. 包含完整的代码

```python"""
    
    print(f"Prompt: {prompt}")
    print("="*50)
    
    # 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("Generated:")
    print(generated)
    print("="*50)
    
    # 测试对话格式
    messages = [
        {"role": "user", "content": "请用PyTorch写一个包含两个全连接层的简单神经网络类，输入维度784，隐藏层256，输出维度10"}
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Chat prompt: {chat_prompt}")
    print("="*50)
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("Generated (chat format):")
    print(generated)

if __name__ == "__main__":
    test_basic_generation()
