import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-Coder-1.3B-Instruct"

print(f"[LOAD] Loading clean base model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

prompt = "Write a simple PyTorch neural network class called Net."

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True
)

print("\n===== BASE MODEL OUTPUT =====\n")
print(response)
print("\n=============================\n")
