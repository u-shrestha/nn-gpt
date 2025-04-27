import json
from pathlib import Path
import pandas as pd
import torch
from ab.nn.api import data as nn_data
from util.Const import out_dir
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments
from util.LLM import LLM
from util.LoRA import LoRA, find_all_linear_names
from util.prompt.NNPrompt import NNPrompt
from util.Const import config_file

class EnhancedModelFinetuner:
    def __init__(self, config_path=config_file):
        self.config = self._load_config(config_path)
        self.output_dir = Path(out_dir) / 'model_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.required_columns = ['nn', 'dataset', 'task', 'epoch', 'accuracy', 'nn_code', 'prm']

    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)
    
    def _standardize_columns(self, df):
        column_map = {
            'nn': ['model', 'network'],
            'dataset': ['data'],
            'epoch': ['epochs'],
            'accuracy': ['acc'],
            'nn_code': ['code', 'model_code'],
            'prm': ['params', 'parameters'],
            'task': ['objective', 'problem', 'task_type', 'target']
        }
        
        lower_columns = [c.lower() for c in df.columns]
        standardized = {}
        
        for std, alts in column_map.items():
            possible = [std.lower()] + [a.lower() for a in alts]
            for idx, col in enumerate(lower_columns):
                if col in possible:
                    standardized[std] = df.columns[idx]
                    break
        
        missing = [c for c in self.required_columns if c not in standardized]
        if missing:
            if 'task' in missing:
                if 'problem_type' in df.columns:
                    df['task'] = df['problem_type']
                    standardized['task'] = 'task'
                    missing.remove('task')
                else:
                    df['task'] = 'classification'
                    standardized['task'] = 'task'
                    missing.remove('task')
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
        return df.rename(columns=standardized)

    def _prepare_training_data(self):
        raw_data = nn_data(only_best_accuracy=False)
        standardized = self._standardize_columns(raw_data)
        
        standardized['epoch'] = pd.to_numeric(standardized['epoch'], errors='coerce').fillna(0).astype(int)
        standardized['accuracy'] = pd.to_numeric(standardized['accuracy'], errors='coerce')
        standardized['prm'] = standardized['prm'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )
        
        standardized_sorted = standardized.sort_values(['nn', 'dataset', 'task', 'epoch'])
        
        # Group by network, dataset, and task
        grouped = standardized_sorted.groupby(['nn', 'dataset', 'task'])
        
        training_rows = []
        
        for group_key, group_data in grouped:
            nn, dataset, task = group_key
            
            # Find epochs 1 and 2 if they exist
            early_epochs = [1, 2]
            for early_epoch in early_epochs:
                early_data = group_data[group_data['epoch'] == early_epoch]
                
                if early_data.empty:
                    continue
                    
                early_row = early_data.iloc[0].to_dict()
                
                # Find maximum accuracy for this group after this early epoch
                later_data = group_data[group_data['epoch'] > early_epoch]
                
                if later_data.empty:
                    max_accuracy = early_row['accuracy']
                    max_epoch = early_row['epoch']
                    max_params = early_row['prm']
                    max_code = early_row['nn_code']
                else:
                    max_accuracy = later_data['accuracy'].max()
                    max_row = later_data[later_data['accuracy'] == max_accuracy].iloc[0].to_dict()
                    max_epoch = max_row['epoch']
                    max_params = max_row['prm']
                    max_code = max_row['nn_code']
                
                # Create a comprehensive training row
                training_row = {
                    'nn': nn,
                    'dataset': dataset,
                    'task': task,
                    'early_epoch': early_epoch,
                    'early_accuracy': early_row['accuracy'],
                    'early_params': early_row['prm'],
                    'max_epoch': max_epoch,
                    'max_accuracy': max_accuracy,
                    'max_params': max_params,
                    'nn_code': max_code  # Store duplicated values only once
                }
                
                training_rows.append(training_row)
        
        merged = pd.DataFrame(training_rows)
        
        # Create prompt and completion for each training row
        merged['prompt'] = (
            "Model: " + merged['nn'].astype(str) + "\n" +
            "Dataset: " + merged['dataset'].astype(str) + "\n" +
            "Task: " + merged['task'].astype(str) + "\n" +
            "Epoch: " + merged['early_epoch'].astype(str) + "\n" +
            "Accuracy: " + merged['early_accuracy'].round(4).astype(str) + "\n" +
            "Hyperparameters: " + merged['early_params']
        )
        
        merged['completion'] = (
            "Max Accuracy: " + merged['max_accuracy'].round(4).astype(str) + "\n" +
            "Final Epoch: " + merged['max_epoch'].astype(str) + "\n" +
            "Optimized Parameters: " + merged['max_params'] + "\n" +
            "Implementation:\n" + merged['nn_code'].astype(str)
        )
        
        output_path = self.output_dir / 'training_data.csv'
        merged.to_csv(output_path, index=False)
        return output_path


    def run(self):
        try:
            training_data_path = self._prepare_training_data()
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model_loader = LLM(
                model_path=self.config['base_model_name'],
                bnb_config=quantization_config,
                access_token=Path(out_dir) / 'token' if self.config.get('token_from_file') else None
            )
            model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()
            
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.config.get('batch_size', 2),
                gradient_accumulation_steps=self.config.get('gradient_accumulation', 4),
                learning_rate=self.config.get('learning_rate', 2e-4),
                num_train_epochs=self.config.get('num_epochs', 3),
                fp16=True,
                logging_dir=str(self.output_dir / 'logs'),
                report_to="none",
                remove_unused_columns=False
            )
            
            trainer = LoRA(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                peft_config=LoraConfig(
                    r=self.config.get('lora_r', 32),
                    lora_alpha=self.config.get('lora_alpha', 64),
                    target_modules=find_all_linear_names(model),
                    lora_dropout=self.config.get('lora_dropout', 0.1),
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            )
            
            trainer.train(
            dataset=NNPrompt(model_loader.get_max_length(), tokenizer).get_dataset(),
            tokenizer=tokenizer,
            output_dir=str(self.output_dir)
            )
            
            trainer.save_model(self.output_dir / 'final_model')
            print(f"Training complete. Results in: {self.output_dir}")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    finetuner = EnhancedModelFinetuner()
    finetuner.run()