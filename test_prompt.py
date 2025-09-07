#!/usr/bin/env python3
"""测试改进的prompt生成质量"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_prompt():
    base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    if torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            trust_remote_code=True, 
            torch_dtype=torch.float32,
            device_map="mps"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            trust_remote_code=True,
        )
    
    # 测试prompt
    test_masked_code = """
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [MASKED LAYER]
        # [MASKED LAYER]
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""
    
    prompt = f"""# [MASKED LAYER]
# [MASKED LAYER]

Answer:
self.conv1 = nn.Conv2d(3, 64, 3)
self.conv2 = nn.Conv2d(64, 128, 3)

# [MASKED LAYER]
# [MASKED LAYER]

Answer:"""
    
    print("=== Testing Prompt ===")
    print(prompt)
    print("\n=== Generating Response ===")
    
    # 生成响应
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(prompt_str, return_tensors="pt")
    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": 64,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 15,
        "temperature": 0.3,
        "repetition_penalty": 1.2,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # 解码响应
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print("Raw response:")
    print(repr(response))
    print("\nFormatted response:")
    print(response)
    
    # 评估响应质量
    lines = response.strip().split('\n')
    valid_lines = []
    invalid_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('self.') and '=' in line and 'nn.' in line:
            valid_lines.append(line)
        else:
            invalid_lines.append(line)
    
    print(f"\n=== Analysis ===")
    print(f"Valid layer definitions: {len(valid_lines)}")
    for line in valid_lines:
        print(f"  ✓ {line}")
    
    print(f"Invalid/extra content: {len(invalid_lines)}")
    for line in invalid_lines:
        print(f"  ✗ {line}")
    
    quality_score = len(valid_lines) / max(1, len(valid_lines) + len(invalid_lines))
    print(f"Quality score: {quality_score:.2f}")

if __name__ == "__main__":
    test_prompt()
