import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from ab.nn.util.Const import out_dir
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer,
    PreTrainedTokenizer, PreTrainedModel
)
from ab.nn.api import data as nn_data
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset
from transformers.trainer_callback import EarlyStoppingCallback

from ab.gpt.util.Const import conf_train_dir, llm_dir, model_dir
from util.Chatbot import ChatBot
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True

def stratified_split(df, seed=42):
    unique_nns = df['nn'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_nns)
    n_train = int(0.8 * len(unique_nns))
    n_val = int(0.1 * len(unique_nns))
    n_test = len(unique_nns) - n_train - n_val
    train_nns = set(unique_nns[:n_train])
    val_nns = set(unique_nns[n_train:n_train + n_val])
    test_nns = set(unique_nns[n_train + n_val:])
    if train_nns & val_nns:
        raise ValueError("Train and validation sets have overlapping neural networks")
    if train_nns & test_nns:
        raise ValueError("Train and test sets have overlapping neural networks")
    if val_nns & test_nns:
        raise ValueError("Validation and test sets have overlapping neural networks")
    return train_nns, val_nns, test_nns

def extract_metrics(text):
    clean = re.sub(r"</?[\w_]+>", "", text)
    acc_matches = re.findall(r'best_accuracy\s*[:=]\s*([0-9.]+)', clean)
    epoch_matches = re.findall(r'best_epoch\s*[:=]\s*([0-9]+)', clean)
    acc = float(acc_matches[-1]) if acc_matches else float('nan')
    epoch = int(epoch_matches[-1]) if epoch_matches else None
    return acc, epoch

class LLMFineTuner:
    PROMPT_PATH = conf_train_dir / 'NN_pre.json'
    MODEL_PATH = "ABrain/HPGPT-DeepSeek-R1-Distill-Qwen-7B-R"

    def __init__(self, output_dir=model_dir(out_dir)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = self._load_prompt_template()
        self.chatbot = None

    def _load_prompt_template(self):
        with open(self.PROMPT_PATH, 'r') as f:
            prompt_dict = json.load(f)
            prompt_lines = prompt_dict['predict_metrics']['prompt']
            # Remove acc_at_max_epoch line from prompt
            prompt_lines = [line for line in prompt_lines if '{acc_at_max_epoch}' not in line]
            return "\n".join(prompt_lines)

    def initialize_chatbot(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.chatbot = ChatBot(model=model, tokenizer=tokenizer, keep_memory=False)

    def prepare_data(self):
        output_csv = self.output_dir / 'nn_results_enhanced_final.csv'
        if output_csv.exists():
            print(f"File '{output_csv}' already exists. Skipping processing.")
            return pd.read_csv(output_csv)

        df = pd.DataFrame(nn_data(only_best_accuracy=False))
        req = ['task', 'dataset', 'metric', 'metric_code', 'nn', 'nn_code', 'duration', 'transform_code', 'prm', 'epoch', 'accuracy']
        missing = set(req) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        idxmax = df.groupby(['task', 'dataset', 'nn'])['accuracy'].idxmax()
        bests = df.loc[idxmax, ['task', 'dataset', 'nn', 'accuracy', 'epoch']]
        bests = bests.rename(columns={'accuracy': 'best_accuracy', 'epoch': 'best_epoch'})

        df = df.merge(bests, on=['task', 'dataset', 'nn'], how='left')
        max_epochs = df.groupby(['task', 'dataset', 'nn'])['epoch'].max().rename('max_epoch')
        df = df.merge(max_epochs, on=['task', 'dataset', 'nn'], how='left')

        # Do NOT add acc_at_max_epoch anywhere

        df['epoch_1_accuracy'] = df[df['epoch'] == 1]['accuracy']
        df['epoch_2_accuracy'] = df[df['epoch'] == 2]['accuracy']
        df['epoch_1_accuracy'] = df.groupby(['task','dataset','nn'])['epoch_1_accuracy'].transform('first')
        df['epoch_2_accuracy'] = df.groupby(['task','dataset','nn'])['epoch_2_accuracy'].transform('first')

        cols = [c for c in [
            'task', 'dataset', 'metric', 'metric_code', 'nn', 'nn_code', 'duration', 'transform_code', 'prm',
            'epoch_1_accuracy', 'epoch_2_accuracy', 'best_accuracy', 'best_epoch', 'max_epoch'
        ] if c in df.columns]
        df[cols].to_csv(output_csv, index=False)
        print(f"Processed and saved to '{output_csv}'")
        return df[cols]

    def format_prompt(self, row):
        return self.prompt_template.format(
            model_code=row.get('nn_code', ''),
            hyperparameters=row.get('prm', ''),
            epoch_1_accuracy=row.get('epoch_1_accuracy', ''),
            epoch_2_accuracy=row.get('epoch_2_accuracy', ''),
            dataset=row.get('dataset', ''),
            task_type=row.get('task', ''),
            transform_code=row.get('transform_code', ''),
            metric_code=row.get('metric_code', ''),
            max_epoch=row.get('max_epoch', ''),
        )

    def format_target(self, row):
        return f"best_accuracy: {row['best_accuracy']}\nbest_epoch: {row['best_epoch']}"

    def prepare_datasets(self):
        df = self.prepare_data()
        train_nns, val_nns, _ = stratified_split(df)
        train_df = df[df['nn'].isin(train_nns)]
        val_df = df[df['nn'].isin(val_nns)]
        def build_texts(df):
            return [
                f"{self.format_prompt(row)}\n\n{self.format_target(row)}"
                for _, row in df.iterrows()
            ]
        return (
            Dataset.from_dict({"text": build_texts(train_df)}),
            Dataset.from_dict({"text": build_texts(val_df)})
        )

    def load_model_and_tokenizer(self):
        print(f"Loading model: {self.MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        self.initialize_chatbot(model, tokenizer)
        print(f"Model loaded. Vocab size: {len(tokenizer)}")
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        return model, tokenizer

    def train(self):
        try:
            model, tokenizer = self.load_model_and_tokenizer()
            train_ds, val_ds = self.prepare_datasets()
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False
            peft_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            def tokenize_fn(examples):
                toks = tokenizer(examples["text"], truncation=True, max_length=384, padding=False)
                toks["labels"] = toks["input_ids"].copy()
                return toks
            train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=4).remove_columns(["text"])
            val_ds = val_ds.map(tokenize_fn, batched=True, num_proc=4).remove_columns(["text"])
            args = TrainingArguments(
                output_dir=str(self.output_dir / 'output'),
                logging_dir=str(self.output_dir / 'logs'),
                num_train_epochs=1,
                warmup_steps=1,
                optim="adamw_torch",
                learning_rate=2e-5,
                logging_steps=10,
                max_grad_norm=1.0,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=8,
                lr_scheduler_type="cosine",
                gradient_checkpointing=True,
                fp16=True,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                weight_decay=0.01,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                group_by_length=True,
                dataloader_num_workers=4,
            )
            collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            print("Starting training...")
            trainer.train()
            final_path = self.output_dir / 'final_model'
            trainer.save_model(str(final_path))
            tokenizer.save_pretrained(str(final_path))
            print(f"Training complete. Model saved to: {final_path}")
        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback; traceback.print_exc(); raise

    def test_and_evaluate(self, num_samples=20, save_path=None):
        df = self.prepare_data()
        train_nns, val_nns, test_nns = stratified_split(df)
        test_df = df[df['nn'].isin(test_nns)]
        model_dir = out_dir / 'finetuned-lora'   # Correct path!
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}. Please train the model first.")
        print(f"Loading fine-tuned model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
        model.eval()

        golds, preds = [], []
        best_epochs = []
        predicted_epochs = []
        prompts, responses = [], []
        all_rows = []
        failed_indices = []

        for i, row in test_df.head(num_samples).iterrows():
            prompt = self.format_prompt(row)
            enc = tokenizer(prompt, return_tensors='pt')
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            gen = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
            pred, pred_epoch = extract_metrics(gen)
            gold = float(row['best_accuracy'])
            print(f"\nSample {i+1}:")
            print(f"Response:\n{gen}\n")
            print(f"<think>{gen}</think>")
            # Collecting everything for saving
            row_data = {
                'task': row.get('task', ''),
                'dataset': row.get('dataset', ''),
                'metric': row.get('metric', ''),
                'metric_code': row.get('metric_code', ''),
                'nn': row.get('nn', ''),
                'epoch1_accuracy': row.get('epoch_1_accuracy', np.nan),
                'epoch2_accuracy': row.get('epoch_2_accuracy', np.nan),
                'duration': row.get('duration', np.nan),
                'prm': row.get('prm', ''),
                'transform_code': row.get('transform_code', ''),
                'best_accuracy': gold,
                'best_epoch': row.get('best_epoch', np.nan),
                'predicted_accuracy': pred,
                'predicted_epoch': pred_epoch,
                'prompt': prompt,
                'response': gen
            }
            all_rows.append(row_data)

            golds.append(gold)
            preds.append(pred)
            best_epochs.append(row.get('best_epoch', np.nan))
            predicted_epochs.append(pred_epoch)
            prompts.append(prompt)
            responses.append(gen)

            if np.isnan(pred):
                print(f"[WARN] Sample {i}: No valid prediction parsed.\nGenerated: '{gen}'\nRow: {row.to_dict()}\n")
                failed_indices.append(i)

        # Save to CSV
        import csv
        results_df = pd.DataFrame(all_rows)
        csv_path = save_path if save_path else (self.output_dir / "test_predictions_detailed.csv")
        results_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nSaved detailed test predictions to: {csv_path}")

        # Metrics
        filtered = [(g, p) for g, p in zip(golds, preds) if not np.isnan(p)]
        if not filtered:
            print("No valid predictions for metric calculation.")
            return
        y_true, y_pred = zip(*filtered)
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        if len(y_true) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
            pearson, _ = pearsonr(y_true, y_pred)
        else:
            pearson = float('nan')

        print(f"\nMetrics on {len(y_true)} test samples:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Pearson correlation: {pearson:.4f}")

        if failed_indices:
            print(f"\n{len(failed_indices)} out of {num_samples} predictions could not be parsed. See warnings above.")

if __name__ == "__main__":
    ft = LLMFineTuner(output_dir= model_dir(out_dir))
    # ft.train()  # Uncomment for training
    ft.test_and_evaluate(num_samples=20) 