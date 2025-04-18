import json
from pathlib import Path
import pandas as pd
import torch
from ab.nn.api import data as nn_data
from ab.nn.util.Const import out_dir
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments
from util.ModelLoader import ModelLoader
from util.LoRATrainer import LoRATrainer, find_all_linear_names
from util.preprocessors.CodePromptPreprocessor import CodePromptPreprocessor

class EnhancedModelFinetuner:
    def __init__(self, config_path=Path(__file__).parent.parent / 'gpt/conf/config.json'):
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
        
        standardized['epoch'] = pd.to_numeric(standardized['epoch'], errors='coerce')
        standardized['accuracy'] = pd.to_numeric(standardized['accuracy'], errors='coerce')
        standardized['prm'] = standardized['prm'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )
        
        max_acc_idx = standardized.groupby(['nn', 'dataset', 'task'])['accuracy'].idxmax()
        max_data = standardized.loc[max_acc_idx]
        
        early_data = standardized[standardized['epoch'].isin([1, 2])]
        if early_data.empty:
            early_data = standardized[standardized['epoch'] == standardized['epoch'].min()]
        
        merged = pd.merge(
            early_data,
            max_data,
            on=['nn', 'dataset', 'task'],
            suffixes=('_early', '_max'),
            how='inner'
        )
        
        merged['prompt'] = (
            "Model: " + merged['nn'].astype(str) + "\n" +
            "Dataset: " + merged['dataset'].astype(str) + "\n" +
            "Task: " + merged['task'].astype(str) + "\n" +
            "Epoch: " + merged['epoch_early'].astype(str) + "\n" +
            "Accuracy: " + merged['accuracy_early'].round(4).astype(str) + "\n" +
            "Hyperparameters: " + merged['prm_early']
        )
        
        merged['completion'] = (
            "Max Accuracy: " + merged['accuracy_max'].round(4).astype(str) + "\n" +
            "Final Epoch: " + merged['epoch_max'].astype(str) + "\n" +
            "Optimized Parameters: " + merged['prm_max'] + "\n" +
            "Implementation:\n" + merged['nn_code_max'].astype(str)
        )
        
        output_path = self.output_dir / 'full_training_data.csv'
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
            
            model_loader = ModelLoader(
                model_path=self.config['base_model_name'],
                bnb_config=quantization_config,
                access_token=Path(out_dir) / 'token' if self.config.get('token_from_file') is not None else None
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
            
            trainer = LoRATrainer(
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
            
            # Pass output_dir explicitly to train()
            trainer.train(
                dataset=CodePromptPreprocessor(
                    max_len=model_loader.get_max_length(),
                    tokenizer=tokenizer
                ).get_dataset(),
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