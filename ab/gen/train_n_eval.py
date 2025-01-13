import json
import os
import shutil
from pathlib import Path

import torch
import torchvision
from transformers import BitsAndBytesConfig, TrainingArguments
from torchvision.transforms import transforms
from peft import LoraConfig

from ab.gen.util.CVModelEvaluator import CVModelEvaluator
from ab.gen.util.LoRATrainer import LoRATrainer, find_all_linear_names
from ab.gen.util.preprocessors.CodePromptPreprocessor import CodePromptPreprocessor
from ab.gen.util.Chatbot import ChatBot
from ab.gen.util.ModelLoader import ModelLoader

access_token = None
token_from_file = False
if token_from_file:
    with open("./token") as f:
        access_token = f.readline()

base_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct" # "meta-llama/CodeLlama-7b-Instruct-hf"
num_epochs = 100
num_test_epochs = 2


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    # Load test prompts
    with open('./util/test_prompts.json') as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    prompts = []
    for val in prompt_dict.values():
        for pr in val:
            prompts.append(pr)

    # Load model and tokenizer
    model_loader = ModelLoader(
        base_model_name,
        bnb_config,
        access_token=access_token,
    )

    model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()

    print("Using Max Length:", model_loader.get_max_length())
    data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer, "./Dataset")
    dataset = data_processor.get_dataset()
    print("Dataset length:", len(dataset))
    ds_updated = False

    print(dataset)

    training_args = TrainingArguments(
        report_to=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    )

    peft_config = LoraConfig(
        r=64,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=find_all_linear_names(model),
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    # data for CV
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform
    )


    # loop train and eval cycles
    for epoch in range(num_epochs):
        out_path = "./Models/epochs64/A" + str(epoch) + "/"

        chat_bot = ChatBot(model, tokenizer)

        # produce new CV models
        for idx, prompt in enumerate(prompts):
            code_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/code.py")
            code_file.parent.mkdir(exist_ok=True, parents=True)
            code = chat_bot.chat(
                prompt
            )
            with open(code_file, 'w') as file:
                file.write(code)

            args_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/args.py")
            args_file.parent.mkdir(exist_ok=True, parents=True)
            with open(args_file, 'w') as file:
                file.write("")

        # evaluate produced CV models
        for cv_model in os.listdir(out_path + "synth_cv_models"):
            cv_model = str(os.fsdecode(cv_model))
            if os.path.isdir(out_path + "synth_cv_models/" + cv_model):
                for tries in range(2):
                    try:
                        evaluator = CVModelEvaluator("Models.epochs.A" + str(epoch) + ".synth_cv_models." + cv_model, train_set, test_set)
                        accuracy = evaluator.evaluate(num_test_epochs)
                        accuracies = {
                            str(evaluator.get_args()): (accuracy, num_test_epochs)
                        }
                        with open(out_path + "synth_cv_models/" + cv_model + "/accuracies.json", "w+") as acc_file:
                            json.dump(accuracies, acc_file)
                        Path("./Dataset/A" + str(epoch) + cv_model).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/code.py", "./Dataset/A" + str(epoch) + cv_model)
                        shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/args.py", "./Dataset/A" + str(epoch) + cv_model)
                        shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/accuracies.json", "./Dataset/A" + str(epoch) + cv_model)
                        ds_updated = True
                        break
                    except Exception as error:
                        print("failed to determine accuracy for", cv_model)
                        with open(out_path + "synth_cv_models/" + cv_model + "/error_" + str(tries) + ".txt", "w+") as error_file:
                            error_file.write(str(error))
                        with open(out_path + "synth_cv_models/" + cv_model + "/code.py", "r") as code_file:
                            code_txt = code_file.read()
                        new_code = chat_bot.chat(
                            'The error "' + str(error) +
                            '" was occurred in the following code. fix this problem. '
                            "Provide only the code. Don't provide any explanation. Remove any text from this reply. + \n " +
                            code_txt,
                            engineer_prompt=False
                        )
                        os.remove(out_path + "synth_cv_models/" + cv_model + "/code.py")
                        with open(out_path + "synth_cv_models/" + cv_model + "/code.py", 'w') as file:
                            file.write(new_code)


        # fine tune model for 1 epoch / Using training_args and save copy
        if ds_updated:
            data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer, "./Dataset")
            dataset = data_processor.get_dataset()
            ds_updated = False

        lora_tuner = LoRATrainer(
            model,
            tokenizer,
            training_args=training_args,
            access_token=access_token,
            peft_config=peft_config
        )
        model = lora_tuner.train(dataset, out_path + base_model_name + "_tuned")

        # del model
        # del tokenizer
        #
        # model_loader = ModelLoader(
        #     base_model_name,
        #     bnb_config,
        #     access_token=access_token,
        #     local_path=out_path + base_model_name + "_tuned"
        # )
        #
        # # use updated model for next round
        # model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()


if __name__ == "__main__":
    main()
