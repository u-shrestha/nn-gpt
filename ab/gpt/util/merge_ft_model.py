import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_path = "finetuned_models/path"  # Folder with adapter_config.json and adapter_model.bin|.safetensors
output_path = "finetuned_models/merged_model_path"

# 1. Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # used one in fine-tuning
    device_map="auto"
)

# 2. Connect LoRA to the Base Model
lora_model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    torch_dtype=torch.float16
)

# 3.  Merge
merged_model = lora_model.merge_and_unload()

# 4. Save
merged_model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("Model successfully saved to: ", output_path)