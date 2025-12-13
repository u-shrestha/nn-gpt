import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

class LocalLLMLoader:
    def __init__(self, model_path, use_quantization=True):
        self.model_path = model_path
        
        # 4-bit Quantization Config (Replicating logic from ab.gpt.util.LLMUtil)
        self.bnb_config = None
        if use_quantization:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print(f"Loading Tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Loading Model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, prompt, max_new_tokens=1024, temperature=0.8, top_k=50, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input prompt from output if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()
