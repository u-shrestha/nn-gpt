#!/usr/bin/env python3

"""
修正后的TuneRL.py - SFT + RL训练，专门针对masked layer补全任务
使用原始TuneRL.py的数据集获取代码
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
from ab.gpt.util.Util import extract_code
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.Const import conf_train_dir, conf_test_dir
try:
    import ab.nn.api as api
    API_AVAILABLE = True
except ImportError:
    print("Warning: ab.nn.api not available, using fallback reward function")
    api = None
    API_AVAILABLE = False
import ast
import json
import re

def reward_fn(completion):
    """计算RL训练的reward - 专门针对masked layer补全"""
    try:
        # 提取代码
        extracted_code = extract_code(completion)
        
        if not extracted_code:
            # 如果extract_code失败，尝试查找自定义层定义
            lines = completion.split('\n')
            code_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('self.') and '=' in stripped:
                    code_lines.append(stripped)
            
            if code_lines:
                extracted_code = '\n'.join(code_lines)
        
        if not extracted_code:
            return 0.1  # 没有提取到代码，给很低的分数
        
        # 基本分数：能提取到代码
        score = 0.3
        
        # 检查是否包含PyTorch层定义的关键词
        pytorch_keywords = ['nn.Conv2d', 'nn.Linear', 'nn.ReLU', 'nn.BatchNorm', 
                          'nn.MaxPool', 'nn.Dropout', 'nn.LSTM', 'nn.Embedding']
        
        for keyword in pytorch_keywords:
            if keyword in extracted_code:
                score += 0.1  # 每个PyTorch关键词增加0.1分
        
        # 检查代码结构
        if 'self.' in extracted_code:
            score += 0.2  # 有正确的self.层定义
        
        if '=' in extracted_code:
            score += 0.1  # 有赋值语句
        
        # 检查是否有完整的层定义（包含参数）
        if '(' in extracted_code and ')' in extracted_code:
            score += 0.2  # 有函数调用格式
        
        # 限制分数在0-1之间
        return min(1.0, max(0.0, score))
        
    except Exception as e:
        print(f"奖励函数计算错误: {e}")
        return 0.1

def compute_reward(prompts, completions, **kwargs):
    """计算RL训练的reward"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        print("="*50)
        print("Raw completion:")
        print(completion)
        print("="*50)
        
        # 提取代码
        extracted_code = extract_code(completion)
        print("Extracted code:")
        print(repr(extracted_code))
        
        # 如果extract_code失败，尝试其他方法
        if extracted_code is None or extracted_code.strip() == "":
            code_patterns = [
                r'```python\s*\n(.*?)\n```',
                r'```\s*\n(.*?)\n```',
                r'self\.[^\n]+nn\.[^\n]+',  # 匹配layer定义
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, completion, re.DOTALL)
                if matches:
                    extracted_code = matches[0] if isinstance(matches[0], str) else '\n'.join(matches)
                    break
        
        if extracted_code is None or extracted_code.strip() == "":
            extracted_code = ""
        
        # 计算reward
        try:
            score = reward_fn(extracted_code) if extracted_code.strip() else 0.0
            print(f"Reward score: {score}")
        except Exception as e:
            print(f"Reward calculation failed: {e}")
            score = 0.0
        
        rewards.append(score)
    
    return rewards

def mask_layer_definitions(code: str) -> tuple[str, str]:
    """将代码中的层定义替换为[MASKED LAYER]，返回masked代码和原始层定义"""
    try:
        lines = code.splitlines()
        masked_lines = list(lines)
        extracted_layers = []

        # 简单的正则表达式匹配nn层定义
        import re
        for i, line in enumerate(lines):
            if re.search(r'self\.\w+\s*=\s*nn\.', line):
                extracted_layers.append(line.strip())
                masked_lines[i] = "        # [MASKED LAYER]"

        return "\n".join(masked_lines), "\n".join(extracted_layers)
    except Exception:
        return code, ""

def format_for_sft(example):
    """格式化数据用于SFT训练 - 修正版本"""
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

def format_for_rl(example):
    """格式化数据用于RL训练 - 从SFT格式转换"""
    if "text" in example:
        # 从SFT格式提取prompt和response
        text = example["text"]
        # 按"Answer:"分割
        parts = text.split("Answer:")
        if len(parts) == 2:
            prompt = parts[0].strip() + " Answer:"
            response = parts[1].strip()
        else:
            # 如果没有"Answer:"，尝试按长度分割
            mid_point = len(text) // 2
            prompt = text[:mid_point]
            response = text[mid_point:]
    else:
        # 兼容旧格式
        prompt = example.get("prompt", "")
        response = example.get("response", "")
    
    return {
        "query": prompt,  # GRPOTrainer期望的字段名
        "response": response.strip(),
    }

def load_original_dataset(tokenizer, base_model):
    """使用原始TuneRL.py的数据获取代码"""
    
    # 配置路径
    train_config_path = conf_test_dir / 'NN_gen.json'
    
    print("正在获取原始数据集...")
    
    # 使用原始的数据处理器
    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=100000)
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)
    dataset.save_to_disk("dataset.json")
    
    print(f"原始数据集大小: {len(dataset)}")
    
    # 大幅减少训练数据量，基于成功的简化SFT测试
    dataset = dataset.select(range(20)).shuffle()  # 只使用20个高质量样本
    

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
        "Answer:"  # 改为Answer:便于后续解析
    ]
    
    def build_masked_dataset(example):
        code = example["response"]
        try:
            parsed = json.loads(code)
            code = parsed["nn_code"]
            masked_model, masked_code = mask_layer_definitions(code)
            
            prompt = "\n".join([
                line if "{masked_prompt}" not in line else line.format(masked_prompt=masked_model)
                for line in PROMPT_TEMPLATE
            ])
            
            # SFT格式：完整的对话文本
            full_text = f"{prompt}\n{masked_code}"
            
            return {
                "text": full_text  # SFTTrainer期望的字段名
            }
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return {
                "text": ""
            }
    
    # 应用masking，创建SFT格式
    sft_dataset = dataset.map(build_masked_dataset, remove_columns=dataset.column_names)
    sft_dataset = sft_dataset.filter(lambda x: x["text"] != "")
    
    print(f"SFT格式数据集大小: {len(sft_dataset)}")
    
    return sft_dataset

def main():
    print("=== SFT + RL训练：Masked Layer补全任务 ===")
    
    # 模型配置
    base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "auto"
    )
    
    # 保守的LoRA配置
    peft_config = LoraConfig(
        r=4,  # 低rank
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # 只训练少数模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # === 加载数据集 ===
    print("\n" + "="*60)
    print("加载数据集（使用原始TuneRL.py逻辑）")
    print("="*60)
    
    # 加载数据集（使用原始TuneRL.py逻辑）
    sft_dataset = load_original_dataset(tokenizer, base_model)
    print(f"SFT Dataset size: {len(sft_dataset)}")
    
    # === SFT训练阶段 ===
    print("\n" + "="*60)
    print("阶段1: SFT训练 - 学习基本的layer补全格式")
    print("="*60)
    
    # SFT训练配置
    sft_training_args = TrainingArguments(
        output_dir="./sft_outputs_masked",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=999,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=False,
        report_to=[]
    )
    
    # SFT训练
    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,  # 使用SFT数据集
        args=sft_training_args,
        peft_config=peft_config
    )
    
    print("开始SFT训练...")
    sft_trainer.train()
    print("SFT训练完成！")
    
    # === 暂时跳过RL训练，专注于测试SFT效果 ===
    print("\n" + "="*60)
    print("阶段2: 测试SFT在masked layer补全任务上的效果")
    print("="*60)
    
    if len(sft_dataset) > 0:
        # 测试第一个样本（从SFT格式转换为测试格式）
        test_text = sft_dataset[0]["text"]
        parts = test_text.split("Answer:")
        if len(parts) == 2:
            test_prompt = parts[0] + "Answer:"
            expected_response = parts[1].strip()
        else:
            test_prompt = test_text[:len(test_text)//2]
            expected_response = test_text[len(test_text)//2:]
            
        print("测试样例:")
        print(test_prompt)
        print("\n预期答案:")
        print(expected_response)
        print("-" * 50)
        
        # 测试模型生成
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print("模型生成结果:")
        print(generated_text)
        
        print("\n提取的层定义:")
        extracted = extract_code(generated_text)
        if extracted:
            print(extracted)
            
            # 尝试评估生成的代码
            try:
                reward_score = reward_fn(extracted)
                print(f"\nReward Score: {reward_score}")
            except Exception as e:
                print(f"\n无法评估代码: {e}")
        else:
            print("未能提取到有效代码")
    
    print("\nSFT训练完成！模型现在应该能够更好地理解layer定义格式。")
    
    # === RL训练阶段 ===
    print("\n" + "="*60)
    print("阶段2: RL训练 - 优化masked layer补全质量")
    print("="*60)
    
    # 转换数据为RL格式
    rl_dataset = sft_dataset.map(format_for_rl, remove_columns=sft_dataset.column_names)
    print(f"RL数据集大小: {len(rl_dataset)}")
    
    if len(rl_dataset) > 0:
        rl_sample = rl_dataset[0]
        print("\n第一个RL样本:")
        print(f"Query: {rl_sample['query'][:200]}...")
        print(f"Response: {rl_sample['response'][:100]}...")
    
    # RL训练配置
    rl_training_args = TrainingArguments(
        output_dir="./rl_outputs_masked",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,  # 更小的学习率用于RL
        logging_steps=1,
        save_steps=999,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=False,
        report_to=[]
    )
    
    try:
        # 暂时跳过复杂的GRPOTrainer，使用简单的迭代优化
        print("使用基于奖励函数的简单优化策略...")
        
        # 生成多个候选答案，选择最好的
        best_score = -1
        best_response = ""
        
        test_query = rl_dataset[0]["query"] if len(rl_dataset) > 0 else "Complete the PyTorch layers:"
        
        for i in range(3):  # 生成3个候选答案
            inputs = tokenizer(test_query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7 + i * 0.1,  # 不同的temperature
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            score = reward_fn(response)
            
            print(f"候选答案 {i+1} (分数: {score:.3f}): {response[:100]}...")
            
            if score > best_score:
                best_score = score
                best_response = response
        
        print(f"最佳候选答案 (分数: {best_score:.3f}):")
        print(best_response)
        print("简单优化完成！")
        
    except ImportError as e:
        print(f"无法导入GRPOTrainer: {e}")
        print("跳过RL训练，使用SFT训练后的模型...")
    except Exception as e:
        print(f"优化过程遇到问题: {e}")
        print("继续使用SFT训练后的模型进行测试...")
    
    # === 最终测试 ===
    print("\n" + "="*60)
    print("最终测试: Masked Layer补全效果")
    print("="*60)
    
    test_masked_prompt = """Complete the missing layers in this PyTorch neural network:

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        [MASKED LAYER]  # 卷积层：输入1通道，输出32通道，3x3卷积核
        [MASKED LAYER]  # ReLU激活
        [MASKED LAYER]  # 最大池化层：2x2
        [MASKED LAYER]  # 全连接层：输入784，输出10
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

Answer:"""
    
    model.eval()
    inputs = tokenizer(test_masked_prompt, return_tensors="pt").to(model.device)
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
    
    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("最终生成结果:")
    print(generated_text)
    
    # 提取并评估代码
    extracted_code = extract_code(generated_text)
    print(f"\n提取的代码:")
    print(extracted_code if extracted_code else "未能提取到有效代码")
    
    # 计算奖励分数
    score = reward_fn(test_masked_prompt + " " + generated_text)
    print(f"\n奖励分数: {score}")
    
    return model
    print("=== 修正后的TuneRL.py ===")
    
    # 模型配置
    base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "auto"
    )
    
    # 保守的LoRA配置
    peft_config = LoraConfig(
        r=4,  # 低rank
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # 只训练少数模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 创建少量高质量训练数据
    training_data = [
        {
            "prompt": "请用PyTorch创建一个ReLU激活层",
            "completion": """```python
import torch
import torch.nn as nn

class ReLULayer(nn.Module):
    def __init__(self):
        super(ReLULayer, self).__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x)
```"""
        },
        {
            "prompt": "请用PyTorch创建一个卷积层，输入通道16，输出通道32，卷积核大小3x3",
            "completion": """```python
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)
```"""
        },
        {
            "prompt": "请用PyTorch创建一个全连接层，输入维度784，输出维度256",
            "completion": """```python
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(784, 256)
    
    def forward(self, x):
        return self.linear(x)
```"""
        }
    ]
    
    dataset = Dataset.from_list(training_data)
    print(f"Training data size: {len(dataset)}")
    
    # 保守的训练配置
    training_args = TrainingArguments(
        output_dir="./sft_outputs_corrected",
        num_train_epochs=1,  # 只训练1轮
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # 很小的学习率
        logging_steps=1,
        save_steps=999,  # 不保存中间checkpoint
        save_strategy="no",
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
    
    print("开始保守的SFT训练...")
    print("修正要点:")
    print("1. 使用3个高质量样本")
    print("2. 保守的LoRA设置 (r=4, 只训练q_proj和v_proj)")
    print("3. 很小的学习率 (1e-5)")
    print("4. 短时间训练 (1个epoch)")
    print("="*50)
    
    trainer.train()
    
    # 测试训练后的效果
    print("\n" + "="*50)
    print("测试SFT训练后的生成效果:")
    print("="*50)
    
    test_prompt = "请用PyTorch创建一个Dropout层，丢弃率0.5"
    print(f"测试prompt: {test_prompt}")
    print("-" * 30)
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("生成结果:")
    print(generated_text)
    
    print("\n" + "="*50)
    print("提取的代码:")
    extracted = extract_code(generated_text)
    if extracted:
        print(extracted)
    else:
        print("未能提取到有效代码")
    
    print("\n修正完成！基于成功的简化SFT方法。")
    print("如果效果良好，可以考虑:")
    print("1. 增加更多高质量训练样本")
    print("2. 进行RL训练")
    print("3. 调整生成参数")

if __name__ == "__main__":
    main()
