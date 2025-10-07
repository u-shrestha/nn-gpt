#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
from ab.gpt.util.Util import extract_code
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.Const import conf_train_dir, conf_test_dir
from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
import ab.nn.api as api

import json
import os

from simple_logger import SimpleCodeLogger

code_logger = SimpleCodeLogger("rl_output")

from typing import Tuple, Any

# ===== Configuration Options =====
LOAD_EXISTING_MODEL = False  # Whether to load saved model
SAVED_MODEL_PATH = "final_model_masked"  # Saved model path
GENERATED_CODE_DIR = "rl_output/generated_models"  # Directory for storing generated models
B_index = 0
# ==================================

def reward_fn(code: str) -> Tuple[float, Any]:
    """Calculate reward and return API result"""
    base_reward = 0.0
    try:
        try:
            # required_snippets = ["class Net", "def forward", "def train_setup", "def learn"]
            # if all(snippet in code for snippet in required_snippets):
            #     base_reward = max(base_reward, 0.2)
            res = api.check_nn(code, task='img-classification', 
                                dataset='cifar-10', metric='acc', 
                                prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
                                    'transform': 'norm_256_flip', 'epoch': 1}, 
                                save_to_db=False)
            code_logger.log_to_file(f"API evaluation result: {res}")
            # res = (model_name, accuracy, accuracy_to_time, res['score'])
            # acc = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else 0.0
            # final_reward = base_reward*0.5 + acc*0.5
            final_reward = res[3]
            return final_reward, res
        except Exception as e:
            code_logger.log_to_file(f"API evaluation failed: {e}")
            return base_reward, None
    except Exception as e:
        code_logger.log_to_file(f"Reward function crashed: {e}")
        return base_reward, None

def compute_reward(prompts, completions, **kwargs):
    global B_index
    rewards = []
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        code_logger.log_to_file("="*50)
        # print(f"Processing sample {i+1}/{len(prompts)}")
        # print(f"Prompt:\n{prompt}\n")
        
        final_code = extract_code(completion)
        if not final_code or final_code.strip() == "":
            final_code = completion
        
        api_result = None
        try:
            if final_code.strip():
                score, api_result = reward_fn(final_code)
            else:
                score = 0.0
            
            code_logger.log_generation(prompt, completion, score, api_result)

            if score > 0:
                out_path = epoch_dir(0)
                model_dir = synth_dir(out_path) / f"B{B_index}"
                code_file = model_dir / new_nn_file
                code_logger.log_to_file(f"[INFO]Saving code to: {code_file}")
                code_file.parent.mkdir(exist_ok=True, parents=True)  # Move here to avoid empty folder
                with open(code_file, 'w') as file:
                    file.write(final_code)
                create_file(model_dir, new_out_file, completion)
                code_logger.log_to_file("Generated successful code:")
                code_logger.log_to_file(final_code)
                B_index += 1
            else:
                print(final_code)
        except Exception as e:
            code_logger.log_to_file(f"Reward calculation failed: {e}")
            score = 0.0
            code_logger.log_generation(prompt, completion, score, None)
            
        # if i > 10:
        #     return rewards
        rewards.append(score)
    return rewards

def mask_layer_definitions(code: str) -> tuple[str, str]:
    try:
        lines = code.splitlines()
        masked_lines = list(lines)
        extracted_layers = []

        import re
        for i, line in enumerate(lines):
            if re.search(r'self\.\w+\s*=\s*nn\.', line):
                extracted_layers.append(line.strip())
                masked_lines[i] = "        # [MASKED LAYER]"

        return "\n".join(masked_lines), "\n".join(extracted_layers)
    except Exception:
        return code, ""
                
def format_for_sft(example, tokenizer, prompt_template):
    code = example["response"]
    try:
        parsed = json.loads(code)
        original_code = parsed["nn_code"]
        
        # Generate masked version
        masked_model, masked_layers = mask_layer_definitions(original_code)
        
        # Build user prompt (show code with masks)
        user_prompt = prompt_template.format(masked_code=masked_model)
        
        # complete_messages = [
        #     {"role": "user", "content": user_prompt},
        #     {"role": "assistant", "content": original_code}
        # ]
        # complete_text = tokenizer.apply_chat_template(
        #     complete_messages,
        #     tokenize=False,
        #     add_generation_prompt=False  # Include complete conversation
        # )
        complete_text = "### Input" + user_prompt + "\n### Response:\n" + original_code
        
        return {
            "prompt": user_prompt,  # Prompt field expected by SFTTrainer
            "completion": original_code,   # Completion field expected by SFTTrainer (complete code as target)
            "text": complete_text,  # Complete conversation text for SFT training
            "original_code": original_code,  # Save original complete code
            "masked_model": masked_model  # Save code with masks
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {
            "prompt": "",
            "completion": "",
            "text": "",
            "original_code": "",
            "masked_model": ""
        }

def format_for_rl(example, tokenizer):
    try:
        user_prompt = example.get("prompt", "")
        
        if not user_prompt:
            return {"prompt": ""}
        
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,   
            tokenize=False                
        )
        
        return {"prompt": prompt_str}
    except Exception as e:
        print(f"Error formatting RL data: {e}")
        return {"prompt": ""}

# def load_original_dataset(test_conf_filename: str = 'NN_alter.json'):
#     cfg_path = conf_test_dir / test_conf_filename
#     with open(cfg_path, 'r', encoding='utf-8') as f:
#         prompt_dict = json.load(f)

#     prompts = []
#     for key in prompt_dict.keys():
#         base_prompt = ""
#         for pr in prompt_dict[key].get('prompt', []):
#             base_prompt += pr + "\n"

#         task = prompt_dict[key]['task']
#         addon_task = prompt_dict[key].get('addon_task')

#         data = api.data(only_best_accuracy=True, task=task).groupby(by="nn").sample(n=1)
#         addon_data = api.data(only_best_accuracy=True, task=addon_task) if addon_task else None

#         for _, row in data.iterrows():
#             para_dict = {}
#             for it in prompt_dict[key].get("input_list", []):
#                 para_dict[it['para']] = row[it['value']]

#             if addon_data is not None and len(addon_data) > 0:
#                 cand = addon_data.loc[addon_data.nn != row['nn']]
#                 if len(cand) > 0:
#                     addon_row = cand.sample(n=1).iloc[0]
#                     for it in prompt_dict[key].get('addon_list', []):
#                         para_dict[it['para']] = addon_row[it['value']]

#             prompts.append((base_prompt.format(**para_dict), row))

#     return prompts

def load_original_dataset(tokenizer):
    train_config_path = conf_test_dir / 'NN_gen.json'
    
    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=100000)
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)

    print(f"Original dataset size: {len(dataset)}")

    dataset = dataset.select(range(1000)).shuffle()
    
    return dataset
    # PROMPT_TEMPLATE = [
    # "Below is a partial implementation of a PyTorch neural network.",
    # "Some layers in the `__init__` method are replaced with the placeholder `[MASKED LAYER]`.",
    # # "Here is the beginning of the implementation:",
    # "",
    # "{masked_prompt}",
    # "",
    # # "Your task is to complete the layer definitions in the `__init__` method. Replace the # [MASKED LAYER], DO NOT change anything else."
    # # "Your task is to generate the layer definitions, DO NOT change anything else."
    # "Your task is to generate the layer definitions."
    # # "Your task is to ONLY replace the [MASKED LAYER] in the `__init__` method."
    # # "Strictly DO NOT change or rewrite any other part of the code.",
    # # "The class name MUST be `Net`",
    # # "The code must include all necessary methods like `__init__`, `forward`, `train_setup`, and `learn`.",
    # "Now output the layer definitions below, No other codes"

    # ]
PROMPT_TEMPLATE = """Below is a partial PyTorch neural network with masked layers:

{masked_code}

Please generate the complete PyTorch neural network code by replacing the [MASKED LAYER] placeholders.
Requirements:
1. Output the complete Python/PyTorch code wrapped in ```python``` code blocks
2. Keep the same class structure (Net class) and method signatures exactly
3. Replace each '# [MASKED LAYER]' with appropriate PyTorch layer definitions
4. NO imports except from torch, NEVER use external libraries like torchvision.models
5. Never change train_setup, learn, forward methods
6. supported_hyperparameters function must remain exactly the same outside the Net class, only have ['lr', 'momentum', 'dropout']
7. Make sure the Net.__init__ method signature matches: __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device)
8. prm only contains 'lr', 'momentum', 'batch', 'transform', 'epoch'
Now generate the complete working code wrapped in ```python``` blocks:"""

def generate_sft_dataset(dataset, tokenizer):
    def build_masked_dataset(example):
        return format_for_sft(example, tokenizer, PROMPT_TEMPLATE)
    sft_dataset = dataset.map(build_masked_dataset, remove_columns=dataset.column_names)
    sft_dataset = sft_dataset.filter(lambda x: x.get("prompt", "") != "" and x.get("completion", "") != "" and x.get("masked_model", "") != "")
    # sft_dataset = sft_dataset.select(range(100))
    return sft_dataset

def main():
    torch.cuda.empty_cache()  
    
    # base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"  
    base_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "22GB"},  
        )
    # Check whether to load saved model
    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading existing model from {SAVED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, SAVED_MODEL_PATH)
        print("Existing model loaded successfully!")
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, 
        # layers_to_transform=list(range(18, 24)),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # === Load Dataset ===
    dataset = load_original_dataset(tokenizer)
    print("Sample original data:")
    print(dataset[0])
    # for idx, prompt in enumerate(dataset.select(range(3))):
    #     prompt, origdf = prompt
    #     print(f"Sample {idx+1} Prompt:\n{prompt}\n")
    #     print(origdf)
    sft_dataset = generate_sft_dataset(dataset, tokenizer)
    print(f"SFT Dataset size: {len(sft_dataset)}")
    # === RL Stage ===
    rl_dataset = sft_dataset.map(lambda x: format_for_rl(x, tokenizer), remove_columns=["text", "completion", "original_code", "masked_model"])
    rl_dataset = rl_dataset.filter(lambda x: x["prompt"] != "")
    print(f"RL dataset size: {len(rl_dataset)}")
    
    grpo_config = GRPOConfig(
        # learning_rate=5e-5,
        learning_rate=1e-4,
        max_completion_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        remove_unused_columns=False,
        logging_steps=10,
        output_dir="./grpo_outputs",
        eval_strategy="no",
        bf16=True,
        # gradient_checkpointing=True,
        num_generations=2,
   
        generation_kwargs={
            "max_new_tokens": 2048,      
            "do_sample": True,
            "top_p": 0.95,              
            "top_k": 50,               
            "temperature": 0.5,        
            "repetition_penalty": 1.1,  
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            # "renormalize_logits": True,
        },
        # use_peft=True
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward, 
        args=grpo_config,
        processing_class=None
    )

    trainer.train()
    
    print("RL training completed!")
    # Save logs and successful codes to rl_output directory
    code_logger.save_log()
    # code_logger.export_success_codes()  
    
    print(f"Saving model to {SAVED_MODEL_PATH}...")
    model.save_pretrained(SAVED_MODEL_PATH)
    print("Model saved successfully!")
    
    return model

if __name__ == "__main__":
    main()
