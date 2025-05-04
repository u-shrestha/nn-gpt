from os import makedirs

import torch
from ab.nn.util.Util import release_memory
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase
)


# When using deepspeed, no Training arguments' initialization after model initialization, if pre-trained model is used
# Reference: https://huggingface.co/docs/transformers/deepspeed?zero-config=ZeRO-3
# With the claim: "The TrainingArguments object must be created before calling the model from_pretrained()"

def find_all_linear_names(model):
    cls = torch.nn.modules.linear.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    return LoraConfig(
        r=32,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True
    )


class LoRA:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 training_args: TrainingArguments,
                 access_token=None,
                 peft_config=None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.access_token = access_token
        if peft_config is None:
            modules = find_all_linear_names(self.model)
            self.peft_config = create_peft_config(modules)
        else:
            self.peft_config = peft_config
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()
        self.peft_model = get_peft_model(self.model, self.peft_config)

    def train(self, dataset: Dataset, tokenizer, output_dir: str):
        self.peft_model.config.use_cache = False
        # Split The dataset
        dataset = dataset.train_test_split(test_size=0.1)

        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        # build the trainer
        print("Parameter configuration of the model")
        print_trainable_parameters(self.peft_model)

        trainer = Trainer(
            model=self.peft_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False)
        )

        # verifying the datatypes before training
        dtypes = {}
        for _, p in self.peft_model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v / total)
        do_train = True

        # starting training
        print("Training...")
        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics(split="train", metrics=metrics)
            trainer.save_metrics(split="train", metrics=metrics)
            trainer.save_state()
            print(metrics)

        # prepare the model for usage
        self.peft_model.config.use_cache = True

        # Saving model
        print("Saving last checkpoint of the model...")
        makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, access_token=self.access_token)

        # Free memory for merging weights
        # del self.model
        # del trainer
        release_memory()
        return self.peft_model
