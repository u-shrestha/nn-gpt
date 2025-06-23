from pprint import pprint
from ab.gpt.util.prompt.NNRLPrompt import NNRLPrompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig
from datasets import Dataset
from ab.gpt.util.NNEval import NNEval
from ab.gpt.util.Const import conf_train_dir
import ab.nn.api as api

def reward_fn(code: str) -> float:
    try:
        res = api.check_nn(code, task='img-classification', 
                           dataset='cifar-10', metric='acc', prm={'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9,
                   'transform': 'norm_256_flip', 'epoch': 1}, save_to_db=False)
        # evaluator = NNEval(code)
        # score = evaluator.evaluate()
        return res[2]
    except Exception as e:
        print("Eval failed", e)
        return 0.0
    
    # reward function
def compute_reward(prompts, completions, **kwargs):
    """
    prompts: List[str]
    completions: List[str]
    Returns: List[float] as rewards
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # print(completion)
        completion = clean_code_text(completion)
        print(completion)
        score = reward_fn(completion)
        print(score)
        rewards.append(score)
    return rewards

def clean_code_text(text):
    lines = text.strip().split('\n')
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return '\n'.join(lines).strip()

# formalize the data
def format_for_rl(example):
    instruction = example["instruction"] if example["instruction"] else ""
    # context = example["context"] if example["context"] else ""
    prompt = instruction.strip()
    response = example["response"] if example["response"] else ""
    response = clean_code_text(response['nn_code'])
    return {
        "prompt": prompt,
        "response": response.strip(),
        "completion": response.strip(),
    }

# configing the model
train_config_path = conf_train_dir / 'Mix copy.json'
base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
# base_model = "ABrain/NNGPT-DeepSeek-Coder-1.3B-Instruct"

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    trust_remote_code=True, 
    quantization_config=bnb_config,
    device_map="auto")

# LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "dense_h_to_4h", "dense_4h_to_h"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # target_modules=["q_proj"],
    # layers_to_transform=list(range(18, 24)),
    # lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# data processing
# data_processor = NNPrompt(100000, tokenizer)

data_processor = NNRLPrompt(100000, tokenizer, train_config_path)
raw_df = data_processor.get_raw_dataset(False, n_training_prompts=None)
clean_df = raw_df[["instruction", "response"]]
dataset = Dataset.from_pandas(clean_df)
dataset.save_to_disk("dataset.json")

dataset = Dataset.load_from_disk("dataset.json")

dataset = dataset.map(format_for_rl, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: len(x["response"]) < 10000)

with open("dataset_dump.txt", "w", encoding="utf-8") as f:
    count = 2
    for i, sample in enumerate(dataset):
        f.write(f"=== Sample {i} ===\n")
        for key, value in sample.items():
            f.write(f"{key}:\n")
            if isinstance(value, str):
                # 多行字符串直接写入（保留换行缩进）
                f.write(value.strip() + "\n")
            else:
                # 非字符串字段转成字符串再写
                f.write(str(value) + "\n")
        count -= 1
        if count <= 0:
            break
print("Dataset size:", len(dataset))
# pprint(dataset[0])

def tokenize_fn(example):
    full_text = example["prompt"] + example["response"]
    return tokenizer(
        full_text,
        max_length=4096,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# SFT
sftDataset = tokenized_dataset.shuffle().select(range(1000))  # Select a subset for SFT
sft_config = SFTConfig(
    output_dir="./sft_outputs",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    bf16=True,
    remove_unused_columns=False
)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sftDataset,
    args=sft_config
)

sft_trainer.train()

# GRPOTrainer
grpo_config = GRPOConfig(
    learning_rate=5e-5,
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
    num_generations=4
    # use_peft=True
)

trainer = GRPOTrainer(
    # config=grpo_config,
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=compute_reward,
    args=grpo_config,
)

trainer.train()
# trainer.save_model("./grpo_outputs/final_checkpoint")