#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.util.Util import extract_str
from ab.gpt.util.Const import conf_train_dir, conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
from ab.gpt.util.Reward import evaluate_code_and_reward
import ab.nn.api as api

import json
import os
import re
import textwrap

from ab.gpt.util.simple_logger import SimpleCodeLogger
from typing import Tuple, Any, List, Dict

# ===== Configuration Options =====
base_model = "/a/mm/out/nngpt/llm/merged_sft_model" # Starting with merged SFT model
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
B_index = 0
# ==================================

def reward_fn(completion: str) -> Dict[str, Any]:
    """Calculate reward based on execution, novelty, and performance."""
    final_code = reconstruct_code(completion)
    if not final_code:
        return {"reward": -1.0, "built_ok": False, "error": "Reconstruction failed (tags missing?)"}

    # 1. Execution & Performance Reward (R_exec, R_perf)
    # Using evaluate_code_and_reward which handles build, forward, train_step, and accuracy
    res = evaluate_code_and_reward(
        final_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
             'transform': 'norm_256_flip', 'epoch': 1},
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_metric_baseline=0.10,
    )
    
    # Base reward components from Reward.py logic
    # r_build, r_forward, r_trainstep, r_metric (accuracy gain)
    
    # 2. Novelty Reward (R_novel) - Simple heuristic for now
    r_novel = 0.0
    init_code = extract_str(completion, '<init>', '</init>')
    if init_code:
        # Check for variety in backbones
        backbone_matches = re.findall(r"TorchVision\(model=['\"]([^'\"]+)['\"]", init_code)
        if len(set(backbone_matches)) >= 2:
            r_novel += 0.2  # Bonus for using different backbones
        
        # Check for FractalUnit usage
        if "FractalUnit" in init_code:
            r_novel += 0.1
            
    res['reward'] += r_novel
    return res

def compute_reward(prompts, completions, **kwargs):
    global B_index
    rewards = []
    
    # Group logic for Novelty (GRPO advantage)
    # We can calculate group-wide statistics here if needed
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        code_logger.log_to_file("="*50)
        torch.cuda.empty_cache() # Clear cache before each evaluation
        
        try:
            res = reward_fn(completion)
            score = res.get('reward', -1.0)
            
            # Log results
            code_logger.log_to_file(f"Batch index {i}, Evaluation result: {res}")
            
            # Save successful models (B_index)
            if res.get('built_ok') and score > 0:
                final_code = reconstruct_code(completion)
                out_path = epoch_dir(0)
                model_dir = synth_dir(out_path) / f"B{B_index}"
                model_dir.mkdir(exist_ok=True, parents=True)
                
                code_file = model_dir / new_nn_file
                with open(code_file, 'w') as f:
                    f.write(final_code)
                
                create_file(model_dir, new_out_file, completion)
                code_logger.log_to_file(f"[INFO] Saved successful code to B{B_index}")
                B_index += 1
                
            code_logger.log_generation(prompt, completion, score, res)
            rewards.append(score)
            
        except Exception as e:
            code_logger.log_to_file(f"Reward calculation failed at index {i}: {e}")
            rewards.append(-1.0)

    return rewards

PROMPT_TEMPLATE = SFTUtil.prompt_template

def load_rl_dataset(tokenizer):
    """Load high-quality Backbone examples for RL."""
    # Use 'rl-bb-test1' prefix as used in SFTGenPrompt
    data = api.data(task='img-classification', nn_prefixes=("rl-bb-test1",))
    if data.empty:
        # Fallback to general classification if no backbone data exists
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = api.data(only_best_accuracy=True, task='img-classification', dataset='cifar-10')

    print(f"Loaded {len(data)} examples for RL")

    prompts = []
    for idx, row in data.iterrows():
        accuracy = row.get('accuracy', 0.8)
        
        user_prompt = PROMPT_TEMPLATE.format(
            accuracy=accuracy, 
            skeleton_code=SFTUtil.skeleton_code, 
            available_patterns=", ".join(SFTUtil.available_patterns), 
            available_backbones=", ".join(SFTUtil.available_backbones)
        )

        messages = [{"role": "user", "content": user_prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        prompts.append({
            "prompt": prompt_str,
            "accuracy": accuracy
        })

    rl_dataset = Dataset.from_list(prompts)
    return rl_dataset.shuffle(seed=42)

def main():
    torch.cuda.empty_cache()  
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load RL dataset (limit for training speed)
    rl_dataset = load_rl_dataset(tokenizer)
    if len(rl_dataset) > 500:
        rl_dataset = rl_dataset.select(range(500))

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model (merged SFT) with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading extra SFT adapter from {SAVED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, SAVED_MODEL_PATH)
        model = model.merge_and_unload()
    
    # Apply LoRA specifically for RL phase
    peft_config = LoraConfig(
        r=16, # Optimized for memory
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads() 

    model.print_trainable_parameters()
    
    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        max_completion_length=1024, # Skeleton blocks are relatively short
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        num_train_epochs=1, # Start with 1 epoch for RL
        remove_unused_columns=False,
        logging_steps=1,
        output_dir="./grpo_backbone_outputs",
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        num_generations=4, # Reduced to save memory while ensuring GRPO functionality
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward, 
        args=grpo_config,
    )
    
    print("Starting GRPO training for Backbone Search...")
    trainer.train()
    
    print(f"Saving model to {SAVED_MODEL_PATH}...")
    model.save_pretrained(SAVED_MODEL_PATH)
    print("Model saved successfully!")
    
    return model

if __name__ == "__main__":
    from ab.gpt.util.simple_logger import SimpleCodeLogger
    from ab.gpt.util.Reward import evaluate_code_and_reward
    from typing import Dict
    
    # Ensure directories exist
    os.makedirs("rl_output", exist_ok=True)
    code_logger = SimpleCodeLogger("rl_output")
    
    main()
