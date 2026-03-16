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
import shutil
import hashlib

from ab.gpt.util.simple_logger import SimpleCodeLogger
from typing import Tuple, Any, List, Dict
from collections import Counter

# 全局频率计数器，用于追踪 Pattern 的分布
pattern_counts = Counter()
history_signatures = []

# ===== Configuration Options =====
base_model = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct" # 使用新的 Backbone 模型
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
B_index = 0
# ==================================

def clean_block(text: str) -> str:
    """Remove common LLM artifacts like markdown code blocks."""
    if not text: return ""
    text = text.strip()
    # Remove ```python ... ```
    text = re.sub(r'^```python\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

def reconstruct_code(completion: str) -> str:
    """Extract XML blocks and reconstruct full code using skeleton."""
    block_code = clean_block(extract_str(completion, '<block>', '</block>'))
    init_code = clean_block(extract_str(completion, '<init>', '</init>'))
    forward_code = clean_block(extract_str(completion, '<forward>', '</forward>'))

    if block_code and init_code and forward_code:
        code = SFTUtil.skeleton_code
        
        # Replace skeleton signatures with LLM-provided blocks (including signatures)
        sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
        code = code.replace(sig_block, textwrap.dedent(block_code))
        
        sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
        code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))
        
        sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
        code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
        return code
    return ""

def extract_structural_fingerprint(init_str: str, forward_str: str) -> str:
    """提取网络连线图级特征，无视变量名小改动"""
    # 1. 抽取实例化字典自我赋值
    type_map = {}
    lines = init_str.split('\n')
    for line in lines:
        match = re.search(r'self\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_\.]+)\(', line)
        if match:
            type_map[match.group(1)] = match.group(2)
            
    # 2. 从 forward 提取操作序列
    seq = []
    calls = re.finditer(r'([a-zA-Z0-9_\.]+)\(', forward_str)
    
    for match in calls:
        call_name = match.group(1)
        if call_name.startswith('self.'):
            var_name = call_name[5:]
            if var_name in type_map:
                seq.append(type_map[var_name])
            else:
                seq.append(f"self.{var_name}")
        elif call_name.startswith(('torch.', 'F.', 'nn.')):
            seq.append(call_name)
        elif call_name in ('adaptive_pool_flatten',):
            seq.append(call_name)
    
    return " -> ".join(seq)

def get_pattern_signature(completion: str) -> str:
    """提取基于调用图序列的鲁棒特征 Hash"""
    init_code = extract_str(completion, '<init>', '</init>')
    forward_code = extract_str(completion, '<forward>', '</forward>')
    
    if not init_code or not forward_code: 
        return "incomplete"
    
    pattern_match = re.search(r"self\.pattern\s*=\s*['\"]([^'\"]+)['\"]", init_code)
    p_name = pattern_match.group(1) if pattern_match else "Unk"
    
    fingerprint = extract_structural_fingerprint(init_code, forward_code)
    sig_hash = hashlib.md5(fingerprint.encode('utf-8')).hexdigest()[:6]
    
    return f"{p_name}_{sig_hash}"

def reward_fn(completion: str, target_pattern: str = None, batch_signatures: List[str] = None) -> Dict[str, Any]:
    """计算基于执行、新颖性和指令遵循的奖励。"""
    final_code = reconstruct_code(completion)
    if not final_code:
        return {"reward": -2.0, "built_ok": False, "error": "Reconstruction failed (tags missing?)"}
        
    forward_code = extract_str(completion, '<forward>', '</forward>')
    if forward_code and "self.pattern" in forward_code:
        return {"reward": -5.0, "built_ok": False, "error": "CHEAT DETECTED: Accessed self.pattern inside forward block"}

    # 1. 基础执行奖励 (R_exec)
    res = evaluate_code_and_reward(
        final_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
             'transform': 'norm_256_flip', 'epoch': 1},
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_metric_baseline=0.05,
    )
    
    # 2. 指令遵循与 Pattern 验证奖励 (R_pattern_instruction)
    r_instruction = 0.0
    sig = get_pattern_signature(completion)
    
    # --- Curriculum Learning Phases ---
    # With 5 epochs and 4 generations per batch, total generations will be 5780.
    # Phase 1 (0-1500 generations): Pure exploration. No strict backbone or target alignment penalties.
    # Phase 2 (1500-3500 generations): Force Backbone Usage. Penalize missing backbones.
    # Phase 3 (>3500 generations): Strict structural formatting. Penalize ignoring the specific prompted pattern name.
    total_history = sum(pattern_counts.values())
    phase = 1
    if total_history > 3500:
        phase = 3
    elif total_history > 1500:
        phase = 2
    
    init_code = extract_str(completion, '<init>', '</init>')
    forward_code = extract_str(completion, '<forward>', '</forward>')
    if init_code and forward_code:
        # Crucial fix: Check structural fingerprint to ensure backbone is ACTUALLY USED, 
        # not just defined in __init__ (Ghost Backbone Exploit).
        fingerprint = extract_structural_fingerprint(init_code, forward_code)
        has_backbone_executed = any(keyword in fingerprint for keyword in ["TorchVision", "torchvision.models", "resnet", "efficientnet", "densenet", "mobilenet", "convnext", "squeezenet", "shufflenet", "mnasnet", "googlenet", "swin"])
        
        if not has_backbone_executed and phase >= 2:
            # Heavily penalize defining but NOT using backbones in forward pass
            r_instruction -= 2.0
            
    if target_pattern:
        if "incomplete" in sig:
            r_instruction -= 0.5  # Moderate penalty: no hardcoded pattern
        elif not sig.startswith(target_pattern):
            # Penalize ignoring the prompt's structural target request.
            if phase >= 3:
                r_instruction -= 1.0  
        else:
            r_instruction += 0.5  # Mild bonus for matching name expectation
    
    # 3. 批次内多样性奖励 (R_batch_diversity)
    r_batch = 0.0
    if batch_signatures and "incomplete" not in sig:
        # Penalize if many others in the SAME batch have the exact same structure hash
        same_count = batch_signatures.count(sig)
        if same_count > 1:
            r_batch = -1.0 * (same_count - 1) # Stronger batch penalty

    # 4. 长期新颖性奖励 (R_pattern_novel)
    r_novel = 0.0
    if res.get('trained_step_ok'):
        # Because we use Hash, EVERY exact structure has a unique signature.
        freq = pattern_counts.get(sig, 0)
        
        # If this exact hash has been generated multiple times, strongly penalize
        if freq > 2:
            r_novel -= 2.0  # Harsh penalty for repeating identical structures
        elif freq == 0:
            r_novel += 5.0  # MASSIVE bonus for the FIRST time seeing this structure! Force exploration!
        else:
            r_novel += 0.5  # Modest bonus for recent discoveries
    else:
        # User highlighted: Encourage exploration! 
        # Do not punish the model just because its novel idea failed to run perfectly.
        r_novel += 0.0
                
    res['reward'] = res.get('reward', 0.0) + r_instruction + r_batch + r_novel
    res['signature'] = sig
    res['target_pattern'] = target_pattern
    return res

def compute_reward(prompts, completions, **kwargs):
    global B_index
    rewards = []

    # 提取本批次的所有签名
    batch_signatures = [get_pattern_signature(c) for c in completions]
    
    # 尝试从 Prompt 中反推 target_pattern 用于奖励计算
    target_patterns = []
    for p in prompts:
        # 兼容带反引号和不带反引号的情况
        match = re.search(r"Target Pattern:\s*`?([^`\s\n*]+)`?", p)
        target_patterns.append(match.group(1) if match else None)

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        code_logger.log_to_file("="*50)
        torch.cuda.empty_cache()

        try:
            target_p = target_patterns[i]
            res = reward_fn(completion, target_pattern=target_p, batch_signatures=batch_signatures)
            score = res.get('reward', -2.0)
            sig = res.get('signature', 'unknown')

            if res.get('built_ok'):
                pattern_counts[sig] += 1

            code_logger.log_to_file(f"Batch index {i}, Target: {target_p}, Signature: {sig}, Result: {res}")

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
                code_logger.log_to_file(f"[INFO] Saved successful code to B{B_index} (Signature: {sig})")
                B_index += 1

            code_logger.log_generation(prompt, completion, score, res)
            rewards.append(score)

        except Exception as e:
            code_logger.log_to_file(f"Reward calculation failed at index {i}: {e}")
            rewards.append(-1.0)

    # 计算多样性指标
    total_valid = sum(pattern_counts.values())
    unique_count = len(pattern_counts)
    
    if total_valid > 0:
        # 主导 Pattern 占比
        most_common_count = pattern_counts.most_common(1)[0][1]
        dominant_share = most_common_count / total_valid
        
        # 计算 Pattern 分布熵 (Entropy)
        import math
        entropy = -sum((count/total_valid) * math.log2(count/total_valid) for count in pattern_counts.values() if count > 0)
    else:
        dominant_share = 0
        entropy = 0

    # 打印 Pattern 统计
    print(f"\n[Pattern Metrics] Unique: {unique_count}, Dominant Share: {dominant_share:.2%}, Entropy: {entropy:.2f}")
    print(f"[Pattern Distribution] Top 5: {dict(pattern_counts.most_common(5))}")
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
    num_patterns = len(SFTUtil.available_patterns)
    
    for idx, row in data.iterrows():
        accuracy = row.get('accuracy', 0.8)
        # 轮询分配 Pattern，确保数据集内部均衡
        target_p = SFTUtil.available_patterns[idx % num_patterns]

        user_prompt = PROMPT_TEMPLATE.format(
            accuracy=accuracy,
            target_pattern=target_p,
            skeleton_code=SFTUtil.skeleton_code, 
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
        r=16, # Optimized further for memory (was 32)
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
        temperature=1.0,  # Lowered from 1.3 to reduce gibberish while maintaining diversity
        learning_rate=5e-5,
        max_completion_length=1024, # Optimized to fit valid code and reduce trailing trash
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        num_train_epochs=5, # Increased from 1 to 5 to allow extensive exploration across curriculum phases
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

    # 清空旧模型目录
    print(f"Cleaning existing models in {epoch_dir()}...")
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    main()