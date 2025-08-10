import argparse

from peft import LoraConfig
from transformers import TrainingArguments

from ab.gpt.util.Const import nngpt_dir
from ab.gpt.util.Tune import tune, ds_conf

use_deepspeed = False

training_args = TrainingArguments(
    report_to=None,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir=nngpt_dir / 'outputs',
    optim='paged_adamw_8bit',
    deepspeed=ds_conf if use_deepspeed else None,
    gradient_checkpointing=True,
)

peft_config = LoraConfig(
    r=32,  # dimension of the updated matrices
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj"
    ],
    layers_to_transform=list(range(18, 24)),
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip', type=int, default=-1, help='Number of epoches to skip the neural network generation.')
    parser.add_argument('-p', '--peft', type=str, default=None, help='Path to saved LoRA layers.')
    args = parser.parse_args()
    tune(3, 1, args.skip, args.peft, 'NN_gen.json', 'NN_gen.json', 'improve_classification_only',
         'nngpt_dsr1_distill_qwen_7b_r.json', training_args, peft_config, n_training_prompt_limit= 40 * 1024)

if __name__ == '__main__':
    main()
