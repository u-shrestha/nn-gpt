import json
import os
import shutil
from pathlib import Path
import deepspeed

import torch
import torchvision
from transformers import BitsAndBytesConfig, TrainingArguments
from torchvision.transforms import transforms
from peft import LoraConfig

from ab.gpt.util.CVModelEvaluator import CVModelEvaluator
from ab.gpt.util.LoRATrainer import LoRATrainer, find_all_linear_names
from ab.gpt.util.preprocessors.CodePromptPreprocessor import CodePromptPreprocessor
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.ModelLoader import ModelLoader

import ab.nn.api as nn_dataset

with open("./conf/config.json") as config_file:
    config = json.load(config_file)
assert isinstance(config, dict)

token_from_file = True if config["token_from_file"] == "True" else False
base_model_name = config["base_model_name"]
num_epochs = int(config["num_epochs"])
num_test_epochs = int(config["num_test_epochs"])
use_deepspeed = True if config["use_deepspeed"] == "True" else False

access_token = None
if token_from_file:
    with open("../../token") as f:
        access_token = f.readline()

# Deepspeed
ds_config = os.path.join("conf","deepspeed_config.json")

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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
        optim="paged_adamw_8bit",
        deepspeed=ds_config if use_deepspeed else None
    )

    # Load test prompts
    with open('./util/test_prompts.json') as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    prompts = []
    for key in prompt_dict.keys():
        # Legency test_prompts handling
        if prompt_dict[key]['single_row']:
            for pr in prompt_dict[key]['prompts']:
                prompts.append(pr)
        else:
            prompt = ""
            for pr in prompt_dict[key]['prompts']:
                prompt+=pr+"\n"
            # Get nn-dataset codes
            if prompt_dict[key]['task']=="all":
                data = nn_dataset.data(only_best_accuracy=True).groupby(by="nn").sample(n=1)
            elif prompt_dict[key]['task']=="":
                data = None
            else:
                data = nn_dataset.data(only_best_accuracy=True,task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
            # Get addon nn-dataset codes
            if prompt_dict[key]['addon_task']=="all":
                addon_data = nn_dataset.data(only_best_accuracy=True).sample(n=1).iloc[0]
            elif prompt_dict[key]['addon_task']=="":
                addon_data = None
            else:
                addon_data = nn_dataset.data(only_best_accuracy=True,task=prompt_dict[key]['addon_task']).sample(n=1).iloc[0]
            if data is None:
                prompts.append(prompt)
            else:
                for _, row in data.iterrows():
                    para_dict = dict()
                    for it in prompt_dict[key]["input_list"]:
                        para_dict[it['para']]=row[it['value']]
                    if not (addon_data is None):
                        for it in prompt_dict[key]["addon_list"]:
                            para_dict[it['para']]=addon_data[it['value']]
                    prompts.append(prompt.format(**para_dict))

    # Load model and tokenizer
    model_loader = ModelLoader(
        base_model_name,
        bnb_config,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        base_path="../../"
    )

    model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()

    # initialize deepspeed before we do infer in ChatBot, since trainer is not initialized now.
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_config)

    print("Using Max Length:", model_loader.get_max_length())
    data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer)
    dataset = data_processor.get_dataset()
    print("Dataset length:", len(dataset))
    ds_updated = False

    peft_config = LoraConfig(
        r=32,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=find_all_linear_names(model),
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    # loop train and eval cycles
    for epoch in range(num_epochs):
        out_path = "../../Models/epochs/A" + str(epoch) + "/"

        chat_bot = ChatBot(model, tokenizer)

        # produce new CV models
        for idx, prompt in enumerate(prompts):
            code_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/code.py")
            code_file.parent.mkdir(exist_ok=True, parents=True)
            code = chat_bot.chat(
                prompt,
                engineer_prompt=True,
                code_only=False,
                max_words=5000
            )
            with open(code_file, 'w') as file:
                file.write(code)

        # evaluate produced CV models
        for cv_model in os.listdir(out_path + "synth_cv_models"):
            cv_model = str(os.fsdecode(cv_model))
            if os.path.isdir(out_path + "synth_cv_models/" + cv_model):
                for tries in range(2):
                    try:
                        evaluator = CVModelEvaluator("../../Models/epochs/A" + str(epoch) + "/synth_cv_models/" + cv_model)
                        accuracy = evaluator.evaluate()
                        accuracies = {
                            # str(evaluator.get_args()): (accuracy, num_test_epochs)
                            str(evaluator.get_args()): accuracy
                        }
                        with open(out_path + "synth_cv_models/" + cv_model + "/accuracies.json", "w+") as acc_file:
                            json.dump(accuracies, acc_file)
                        Path("./Dataset/A" + str(epoch) + cv_model).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/code.py", "./Dataset/A" + str(epoch) + cv_model)
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
            data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer)
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
