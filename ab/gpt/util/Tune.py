import json
import os
import shutil
import traceback
from os.path import isfile
from pathlib import Path

import ab.nn.api as lemur
import deepspeed
import pandas as pd
from ab.nn.util.Const import ab_root_path
from ab.nn.util.Util import release_memory, crate_file
from transformers import TrainingArguments
from peft import (LoraConfig, PeftModel, prepare_model_for_kbit_training)
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import hp_file, conf_dir, conf_train_dir, conf_test_dir, conf_llm_dir, epoch_dir, new_nn_file, nngpt_dir, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.NNEval import NNEval
from ab.gpt.util.Util import nn_accepted, verify_nn_code, exists
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

ds_conf = conf_dir / 'DeepSpeed.json'


def tune(test_nn, nn_epoch, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)
    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    token_from_file = config['token_from_file']
    base_model_name = config['base_model_name']
    llm_tune_epochs = int(config['num_epochs'])
    use_deepspeed = config['use_deepspeed']
    only_best_accuracy = config['only_best_accuracy']
    context_length = config.get('context_length')

    access_token = None
    if token_from_file:
        with open(ab_root_path / 'token') as f:
            access_token = f.readline()

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
    )

    peft_config = LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj"
        ],
        layers_to_transform=list(range(19, 24)),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f'[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {llm_path}')
    train_config_path = conf_train_dir / llm_tune_conf

    # Load test prompts
    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    # Load model and tokenizer
    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    # print(model)
    if not (llm_path is None):
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path)
        model = model.merge_and_unload()

    # initialize deepspeed before we do infer in ChatBot, since trainer is not initialized now.
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    print('Using Max Length:', model_loader.get_max_length())
    data_processor = NNGenPrompt(model_loader.get_max_length(), tokenizer, train_config_path)
    dataset = data_processor.get_dataset(only_best_accuracy)
    print('Dataset length:', len(dataset))

    already_peft = False  # Prevent multiple LoRA layers
    # loop train and eval cycles
    chat_bot = ChatBot(model, tokenizer)  # Only initialize ONCE

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        # Move inside the loop to create new prompt with newly created models.
        print('Preparing prompts for generation, this might take a while...')
        prompts = []
        for key in conf_keys:
            if epoch < skip_epoch:
                continue  # Prompts are useless when generation is skipped
            prompt = ''
            for pr in prompt_dict[key]['prompts']:
                prompt += pr + '\n'
            # Get nn-dataset codes
            data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by='nn').sample(n=1)[:test_nn]
            # Get addon nn-dataset codes
            addon_data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]
                ## Avoid sampling the same nn_code
                addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
                for it in prompt_dict[key]['addon_list']:
                    para_dict[it['para']] = addon_row[it['value']]
                prompts.append((prompt.format(**para_dict), row))

        from tqdm import tqdm
        # produce new CV models
        models_dir = synth_dir(out_path)
        # print(f"prompts: {prompts}")
        for idx, prompt in tqdm(enumerate(prompts)):
            if epoch < skip_epoch:
                print(f'Skipped Epoch {epoch}')
                continue  # Skipped
            model_dir = models_dir / f'B{idx}'
            prompt, origdf = prompt
            code, hp, full_out = chat_bot.chat(
                prompt,
                engineer_prompt=True,
                code_only=True,
                max_words=5000  ## Reduce memory usage
            )
            print(f'1 gen hyperparams: {hp}')
            crate_file(model_dir, hp_file, hp)
            crate_file(model_dir, new_nn_file, code)
            crate_file(model_dir, new_out_file, full_out)
            df_file = model_dir / 'dataframe.df'
            if origdf is None:
                if isfile(df_file):  # Clean up dataframe.df, if no additional information generated this time.
                    os.remove(df_file)
                    print(f'[DEBUG]Removed unmatched file: {df_file}')
            else:
                crate_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                # Store DataFrame information, mainly for passing parameters to evaluator.
                origdf.to_pickle(df_file)
        release_memory()
        # evaluate produced CV models
        if exists(models_dir):
            for cv_model in os.listdir(models_dir):
                cv_model = str(os.fsdecode(cv_model))
                gen_nn_dir = models_dir / cv_model
                code_file = gen_nn_dir / new_nn_file
                if verify_nn_code(gen_nn_dir, code_file):
                    for tries in range(2):
                        try:
                            df = None
                            df_file = gen_nn_dir / 'dataframe.df'
                            if isfile(df_file):  # The code has additional information provided
                                df = pd.read_pickle(df_file)
                                prm = df['prm']
                                # prm['epoch'] = df['epoch']
                                prm['epoch'] = nn_epoch
                                print(f'Determining {cv_model} with prm{prm}')
                                print(f'Model Path:{gen_nn_dir}')
                                evaluator = NNEval(gen_nn_dir, task=df['task'], dataset=df['dataset'], metric=df['metric'], prm=prm, save_to_db=True,
                                                   prefix=df['nn'].split('-')[0], save_path=gen_nn_dir)
                            else:
                                evaluator = NNEval(gen_nn_dir)
                            eval_results = evaluator.evaluate(code_file)
                            eval_info = {str(evaluator.get_args()): eval_results}
                            name, new_accuracy, code_quality = eval_results
                            with open(gen_nn_dir / 'eval_info.json', 'w+') as f:
                                json.dump(eval_info, f)
                            if nn_accepted(gen_nn_dir):
                                dataset_dir = nngpt_dir / 'new_lemur'
                                nn_dir = dataset_dir / 'nn'
                                stat_dir = dataset_dir / 'stat'
                                Path(nn_dir).mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(gen_nn_dir / new_nn_file, nn_dir / f'{name}.py')
                                nn_model_dir = stat_dir / name
                                if df is None:
                                    Path(nn_model_dir).mkdir(parents=True, exist_ok=True)
                                    for epo in range(prm['epoch']):
                                        f_nm = f'{epo + 1}.json'
                                        shutil.copyfile(gen_nn_dir / f_nm, nn_model_dir / f_nm)
                                else:
                                    dr_nm = stat_dir / f"{df['task']}_{df['dataset']}_{df['metric']}_{name}"
                                    Path(dr_nm).mkdir(parents=True, exist_ok=True)
                                    for epo in range(prm['epoch']):
                                        f_nm = f'{epo + 1}.json'
                                        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)
                                break
                        except Exception as error:
                            print('failed to determine accuracy for', cv_model)
                            crate_file(gen_nn_dir, f'error_{tries}.txt', str(traceback.format_exc()))  # Track traceback to have rich information on the exception.
                            with open(code_file, 'r') as f:
                                code_txt = f.read()
                            release_memory()
                            try:
                                new_code, new_hp, full_out = chat_bot.chat(
                                    'The error "' + str(error) +
                                    '" was occurred in the following code. fix this problem. '
                                    "Provide only the code. Don't provide any explanation. Remove any text from this reply. + \n " +
                                    code_txt,
                                    engineer_prompt=False,
                                    max_words=5000)
                                print(f'2 gen hyperparams: {new_hp}')
                                crate_file(gen_nn_dir, hp_file, new_hp)
                                crate_file(gen_nn_dir, new_nn_file, new_code)
                                crate_file(gen_nn_dir, new_out_file, full_out)
                            except:
                                pass
                            release_memory()
        print('[DEBUG] Release_memory.')
        release_memory()
        print('Clear LEMUR query cache.')
        lemur.data.cache_clear()
        print('The cache has been cleared.')
        # fine tune model for 1 epoch / Using training_args and save copy
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        model.train()
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        lora_tuner = LoRA(
            model,
            tokenizer,
            training_args=training_args,
            access_token=access_token,
            already_peft=already_peft,
            peft_config=peft_config)
        already_peft = True
        model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
