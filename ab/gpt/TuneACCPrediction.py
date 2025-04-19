import json
from os import makedirs

import torch
from ab.nn.api import data as data
from ab.gpt.util.Const import acgpt_dir
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments

from ab.gpt.util.Const import config_file
from util import LLM, LoRA
from util.LoRA import LoRA, find_all_linear_names
from util.LLM import LLM
from util.prompt.CodePromptPreprocessor import CodePromptPreprocessor


# todo: This is a specific fine-tuning implementation by Yashkumar Dhameliya and Yash Kathiriya, expected to be merged into the common pipeline.

class ModelFinetuner:
    def __init__(self, config_path=config_file):
        self.config = self._load_config(config_path)
        self.output_dir = acgpt_dir / 'model_results'
        makedirs(self.output_dir, exist_ok=True)
        
    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)
            
    def _get_top_models(self):
        """Get models by accuracy for each architecture"""
        df = data(only_best_accuracy=True)
        # Handle the grouping to get top models per architecture
        architectures = df['nn'].unique()
        result_dict = {}
        for arch in architectures:
            subset = df[df['nn'] == arch]
            result_dict[arch] = subset.nlargest(2, 'accuracy')
        return result_dict
        
    def _save_model_params(self, model_name, params):
        """Save parameters to model-specific JSON file"""
        output_file = self.output_dir / f"{model_name}.json"
        with open(output_file, 'w') as f:
            json.dump(params, f, indent=2)
            
    def _get_training_data(self, model_df):
        """Create standardized training examples for a specific model"""
        return [
            {
                'prompt': f"Generate {row['nn']} for {row['task']} on {row['dataset']} "
                         f"with params: {row['prm']}",
                'code': row['nn_code'],
                'metadata': {
                    k: row[k] for k in [
                        'task', 'dataset', 'metric', 'metric_code',
                        'nn', 'epoch', 'accuracy', 'duration', 'prm'
                    ]
                }
            }
            for _, row in model_df.iterrows()
        ]
        
    def run(self):
        # Load base model
        model_loader = LLM(
            self.config['base_model_name'],
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            access_token=acgpt_dir / 'token' if self.config['token_from_file'] is not None else None)
        model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()
        max_len = model_loader.get_max_length()
        
        # Get top models grouped by architecture
        models_by_architecture = self._get_top_models()
        
        # Train each architecture separately
        for arch_name, model_df in models_by_architecture.items():
            print(f"Processing architecture: {arch_name}")
            
            # Save model parameters
            model_params = model_df.to_dict('records')
            self._save_model_params(arch_name, model_params)
            
            # Generate training data for this architecture
            training_data = self._get_training_data(model_df)
            
            # Create and setup the preprocessor for this architecture
            preprocessor = CodePromptPreprocessor(
                max_len=max_len,
                tokenizer=tokenizer
            )
            
            # Set examples on the preprocessor if it has an examples attribute
            if hasattr(preprocessor, 'examples'):
                preprocessor.examples = training_data
            else:
                # If no examples attribute, check if the class has a method to add examples
                if hasattr(preprocessor, 'add_examples'):
                    preprocessor.add_examples(training_data)
                else:
                    # Otherwise, look for any method that might accept the examples
                    for method_name in dir(preprocessor):
                        if 'example' in method_name.lower() and callable(getattr(preprocessor, method_name)):
                            try:
                                getattr(preprocessor, method_name)(training_data)
                                break
                            except:
                                continue
            
            try:
                # Get the dataset
                dataset = preprocessor.get_dataset()
                
                # Configure model-specific output directory
                model_output_dir = acgpt_dir / f"finetuned_models/{arch_name}"
                
                # Configure and run training for this architecture
                trainer = LoRA(
                    model,
                    tokenizer,
                    training_args=TrainingArguments(
                        output_dir=model_output_dir,
                        per_device_train_batch_size=2,
                        gradient_accumulation_steps=4,
                        learning_rate=2e-4,
                        num_train_epochs=self.config['num_epochs'],
                        fp16=True
                    ),
                    peft_config=LoraConfig(
                        r=32,
                        lora_alpha=64,
                        target_modules=find_all_linear_names(model),
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                )
                
                trainer.train(dataset, output_dir="finetuned_models")
                
                # Save the model for this architecture
                trainer.save_model(model_output_dir / 'final')
                print(f"Finished training for architecture: {arch_name}")
                
            except Exception as e:
                print(f"Error training architecture {arch_name}: {str(e)}")
                continue
        
if __name__ == "__main__":
    finetuner = ModelFinetuner()
    finetuner.run()