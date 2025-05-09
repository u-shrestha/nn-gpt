import json
from pathlib import Path
import pandas as pd
import os
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from ab.nn.api import data as nn_data
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from datasets import Dataset
from trl import SFTTrainer
from transformers.trainer_callback import EarlyStoppingCallback

class LLMFineTuner:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get("output_dir", "model_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.required_columns = ['nn', 'dataset', 'task', 'epoch', 'accuracy', 'nn_code', 'prm']
        
    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)
    
    def prepare_data(self):
        OUTPUT_CSV = self.output_dir / 'nn_results_enhanced_final.csv'
        
        if os.path.exists(OUTPUT_CSV):
            print(f"File '{OUTPUT_CSV}' already exists. Skipping processing.")
            return pd.read_csv(OUTPUT_CSV)

        raw_data = nn_data(only_best_accuracy=False)  # Assuming nn_data is defined elsewhere
        df = pd.DataFrame(raw_data)

        required_columns = ['task', 'dataset', 'metric', 'metric_code', 'nn', 'nn_code',
                          'duration', 'transform_code', 'prm', 'epoch', 'accuracy']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df['epoch_1_accuracy'] = pd.NA
        df['epoch_2_accuracy'] = pd.NA
        df['best_accuracy'] = pd.NA
        df['best_epoch'] = pd.NA
        df['max_epoch'] = pd.NA

        unique_combinations = df[['task', 'dataset', 'nn']].drop_duplicates()
        print(f"Found {len(unique_combinations)} unique task-dataset-model combinations")

        for _, combo in unique_combinations.iterrows():
            task, dataset, model = combo['task'], combo['dataset'], combo['nn']
            group = df[(df['task'] == task) &
                     (df['dataset'] == dataset) &
                     (df['nn'] == model)]

            if group.empty:
                continue

            max_acc = group['accuracy'].max()
            best_row = group[group['accuracy'] == max_acc].iloc[0]
            best_epoch = best_row['epoch']

            epoch1_acc = group[group['epoch'] == 1]['accuracy']
            epoch2_acc = group[group['epoch'] == 2]['accuracy']
            max_epoch = group['epoch'].max()

            mask = ((df['task'] == task) &
                   (df['dataset'] == dataset) &
                   (df['nn'] == model))

            df.loc[mask, 'best_accuracy'] = max_acc
            df.loc[mask, 'best_epoch'] = best_epoch
            df.loc[mask, 'max_epoch'] = max_epoch

            if not epoch1_acc.empty:
                df.loc[mask, 'epoch_1_accuracy'] = epoch1_acc.values[0]
            if not epoch2_acc.empty:
                df.loc[mask, 'epoch_2_accuracy'] = epoch2_acc.values[0]

        output_columns = ['task', 'dataset', 'metric', 'metric_code', 'nn', 'nn_code',
                        'duration', 'transform_code', 'prm',
                        'epoch_1_accuracy', 'epoch_2_accuracy',
                        'best_accuracy', 'best_epoch', 'max_epoch']
        final_columns = [col for col in output_columns if col in df.columns]
        df_final = df[final_columns]

        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully processed and saved to '{OUTPUT_CSV}'")
        return df_final
    
    def stratified_split(self, df, group_cols=['task', 'dataset', 'nn'], seed=42):
        train_list, val_list, test_list = [], [], []
        rng = np.random.default_rng(seed)

        grouped = df.groupby(group_cols)

        for _, group in grouped:
            group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
            total = len(group)
            n_train = int(round(0.8 * total))
            n_val = int(round(0.1 * total))
            n_test = total - n_train - n_val

            train_list.append(group.iloc[:n_train])
            val_list.append(group.iloc[n_train:n_train + n_val])
            test_list.append(group.iloc[n_train + n_val:])

        train_df = pd.concat(train_list).reset_index(drop=True)
        val_df = pd.concat(val_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        return train_df, val_df, test_df
    
    def generate_prompt(self, row):
        return f"""Task: {row['task']}
Dataset: {row['dataset']}
Model: {row['nn']}
Best Accuracy: {row['best_accuracy']:.4f}
Best Epoch: {row['best_epoch']}
Epoch 1 Accuracy: {row.get('epoch_1_accuracy', 'N/A')}
Epoch 2 Accuracy: {row.get('epoch_2_accuracy', 'N/A')}
Parameters: {row.get('prm', 'N/A')}"""
    
    def prepare_datasets(self):
        df = self.prepare_data()
        train_df, val_df, _ = self.stratified_split(df)
        
        # Generate prompts for each dataset
        train_df['text'] = train_df.apply(self.generate_prompt, axis=1)
        val_df['text'] = val_df.apply(self.generate_prompt, axis=1)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['text']])
        val_dataset = Dataset.from_pandas(val_df[['text']])
        
        return train_dataset, val_dataset
    
    def load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model_name"],
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.config["base_model_name"])
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def train(self):
        try:
            model, tokenizer = self.load_model_and_tokenizer()
            train_dataset, val_dataset = self.prepare_datasets()
            
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False

            peft_config = LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, peft_config)

            # Tokenize the datasets
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)

            training_args = TrainingArguments(
                output_dir=str(self.output_dir / 'output'),
                logging_dir=str(self.output_dir / 'logs'),
                num_train_epochs=35,
                warmup_steps=100,
                optim="adamw_torch",
                learning_rate=1e-5,
                logging_steps=10,
                max_grad_norm=1.0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                lr_scheduler_type="cosine",
                gradient_checkpointing=True,
                fp16=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                weight_decay=0.01,
                save_total_limit=3,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer, 
                    pad_to_multiple_of=8, 
                    return_tensors="pt", 
                    mlm=False
                ),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            trainer.train()
            trainer.save_model(str(self.output_dir / 'final_model'))
            print(f"Training complete. Model saved to: {self.output_dir / 'final_model'}")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Create a proper config file first
    config_data = {
        "base_model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "output_dir": "./output"
    }
    
    config_file = "./ab/gpt/conf/llm/ds_coder_1.3b_instruct.json"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # Write the actual config data
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)
    
    finetuner = LLMFineTuner(config_file)
    finetuner.train()