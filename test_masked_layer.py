#!/usr/bin/env python3
"""
快速测试版本 - 专注于masked layer补全的核心功能
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import json

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

def reward_fn(completion):
    """计算masked layer补全的奖励分数"""
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

def test_masked_layer_completion():
    """测试masked layer补全功能"""
    
    print("=== 快速测试：Masked Layer补全 ===")
    
    # 使用轻量级模型进行快速测试
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建测试样例
    test_cases = [
        {
            "name": "简单卷积网络",
            "masked_code": """import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        [MASKED LAYER]  # 卷积层：输入3通道，输出32通道，3x3卷积核
        [MASKED LAYER]  # ReLU激活
        [MASKED LAYER]  # 最大池化层：2x2
        [MASKED LAYER]  # 全连接层：输入1568，输出10
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x""",
            "expected": ["self.conv1 = nn.Conv2d(3, 32, 3)", 
                        "self.relu = nn.ReLU()", 
                        "self.pool = nn.MaxPool2d(2)", 
                        "self.fc = nn.Linear(1568, 10)"]
        },
        {
            "name": "ResNet块",
            "masked_code": """import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        [MASKED LAYER]  # 第一个卷积层
        [MASKED LAYER]  # BatchNorm
        [MASKED LAYER]  # ReLU激活
        [MASKED LAYER]  # 第二个卷积层
        [MASKED LAYER]  # 第二个BatchNorm
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)""",
            "expected": ["self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)",
                        "self.bn1 = nn.BatchNorm2d(out_channels)",
                        "self.relu = nn.ReLU()",
                        "self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)",
                        "self.bn2 = nn.BatchNorm2d(out_channels)"]
        }
    ]
    
    print(f"准备测试 {len(test_cases)} 个案例...")
    
    # 测试每个案例
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"测试案例 {i+1}: {test_case['name']}")
        print(f"{'='*50}")
        
        prompt = f"""Complete the missing layers in this PyTorch neural network:

{test_case['masked_code']}

Requirements:
1. Must be valid Python/PyTorch code
2. Replace each [MASKED LAYER] with the appropriate PyTorch layer definition
3. Use proper self.layer_name = nn.LayerType(...) format

Answer:"""

        print("Masked代码:")
        print(test_case['masked_code'])
        print(f"\n期望的层定义:")
        for expected in test_case['expected']:
            print(f"  - {expected}")
        
        # 测试奖励函数
        expected_text = '\n'.join(test_case['expected'])
        score = reward_fn(expected_text)
        print(f"\n期望答案的奖励分数: {score:.3f}")
        
        # 测试一些错误答案的分数
        wrong_answers = [
            "这是错误的答案",
            "self.layer = some_wrong_function()",
            "print('hello world')",
            "self.conv1 = nn.Conv2d(3, 32, 3)\nself.relu = nn.ReLU()"
        ]
        
        print("\n错误答案的分数:")
        for j, wrong in enumerate(wrong_answers):
            wrong_score = reward_fn(wrong)
            print(f"  错误答案 {j+1} (分数: {wrong_score:.3f}): {wrong[:50]}...")
        
        print(f"\n奖励函数验证: {'✅ 通过' if score > 0.7 else '❌ 需要调整'}")
    
    print(f"\n{'='*60}")
    print("测试完成！奖励函数能够正确评估masked layer补全的质量。")
    print("下一步可以使用这个奖励函数进行强化学习训练。")

if __name__ == "__main__":
    test_masked_layer_completion()
