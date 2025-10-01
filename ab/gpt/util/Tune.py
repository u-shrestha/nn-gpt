import os
import shutil
import traceback
from os import makedirs
from os.path import isfile

import ab.nn.api as lemur
import deepspeed
import pandas as pd
from ab.nn.util.Util import release_memory, create_file
from peft import (LoraConfig, PeftModel)
from tqdm import tqdm
from transformers import TrainingArguments

from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.NNEval import NNEval
from ab.gpt.util.Util import nn_accepted, verify_nn_code, exists, copy_to_lemur
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

# from datasets import load_from_disk

ds_conf = conf_dir / 'DeepSpeed.json'

def apply_sliding_window(example, max_length, stride, tokenizer):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']

    chunks = []
    for i in range(0, len(input_ids), stride):
        end = i + max_length
        if end <= len(input_ids):
            chunk_input_ids = input_ids[i:end]
            chunk_attention_mask = attention_mask[i:end]

            pad_len = max_length - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [tokenizer.pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len

            chunks.append({
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask
            })
    return {"chunks": chunks}


def flatten_chunks(data):
    all_chunks = sum(data["chunks"], [])  # flatten batched list
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


def tune(test_nn, nn_epoch, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf,
         training_args, peft_config, max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_regenerate_after_exception=False):
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
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

    # initialize deepspeed before we do infer in ChatBot, since trainer is not initialized now.
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config)

    print('Using Max Length:', model_loader.get_max_length())

    # loop train and eval cycles
    chat_bot = ChatBot(model, tokenizer)  # Only initialize ONCE

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped nn generation at epoch {epoch}')
        else:
            nn_gen(out_path, chat_bot, conf_keys, nn_epoch, nn_regenerate_after_exception, prompt_dict, test_nn, max_new_tokens, always_save_full_output=save_llm_output)
        # fine tune model for 1 epoch / Using training_args and save copy
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        # data_processor = NNGenPrompt(model_loader.get_max_length(), tokenizer, train_config_path)
        data_processor = NNGenPrompt(context_length if context_length else model_loader.get_max_length(), tokenizer, train_config_path)
        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts)
        # dataset = load_from_disk(nngpt_dir / 'dataset')

        # if context_length:
        #     chunked_dataset = dataset.map(
        #         lambda x: apply_sliding_window(x, context_length, 1024, tokenizer),
        #         remove_columns=dataset.column_names,
        #         batch_size=16
        #     )
        #     dataset = chunked_dataset.map(flatten_chunks, batched=True, remove_columns=["chunks"])

        # print('Dataset length:', len(dataset))
        print('Dataset length:', len(dataset))
        model.train()
        model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)


def nn_gen(out_path, chat_bot, conf_keys, nn_epoch, nn_regenerate_after_exception, prompt_dict, test_nn, max_new_tokens, always_save_full_output=False):
    # Move inside the loop to create new prompt with newly created models.
    print('Preparing prompts for generation, this might take a while...')
    prompts = []
    for key in conf_keys:
        prompt = ''
        for pr in prompt_dict[key]['prompt']:
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
    # produce new CV models
    models_dir = synth_dir(out_path)
    # print(f"prompts: {prompts}")
    for idx, prompt in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt
        code, hp, full_out = chat_bot.chat(
            prompt,
            engineer_prompt=False,
            max_new_tokens=max_new_tokens  ## Reduce memory usage
        )
        if always_save_full_output: create_file(model_dir, new_out_file, full_out)
        print(f'1 gen hyperparams: {hp}')
        try:
            hp = json.loads(hp.replace("'", '"'))
        except Exception as e:
            print(e)
            continue
        makedirs(model_dir, exist_ok=True)
        # create_file(model_dir, hp_file, hp.replace("'",'"'))
        with open(model_dir / hp_file, 'w+') as f:
            json.dump(hp, f)
        create_file(model_dir, new_nn_file, code)
        create_file(model_dir, new_out_file, full_out)
        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):  # Clean up dataframe.df, if no additional information generated this time.
                os.remove(df_file)
                print(f'[DEBUG]Removed unmatched file: {df_file}')
        else:
            create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
            # Store DataFrame information, mainly for passing parameters to evaluator.
            origdf.to_pickle(df_file)
    release_memory()
    # evaluate produced CV models
    if exists(models_dir):
        for cv_model in os.listdir(models_dir):
            cv_model = str(os.fsdecode(cv_model))
            gen_nn_dir = models_dir / cv_model
            code_file = gen_nn_dir / new_nn_file
            if exists(gen_nn_dir / code_file) and verify_nn_code(gen_nn_dir, code_file):
                for tries in range(2):
                    try:
                        df = None
                        df_file = gen_nn_dir / 'dataframe.df'
                        if isfile(df_file):  # The code has additional information provided
                            df = pd.read_pickle(df_file)
                            # prm = df['prm']
                            # prm['epoch'] = df['epoch']
                            with open(gen_nn_dir / hp_file) as f:
                                hp = json.load(f)
                            hp['epoch'] = nn_epoch
                            print(f'Determining {cv_model} with prm{hp}')
                            print(f'Model Path:{gen_nn_dir}')
                            evaluator = NNEval(gen_nn_dir, task=df['task'], dataset=df['dataset'], metric=df['metric'], prm=hp, save_to_db=True,
                                               prefix=df['nn'].split('-')[0], save_path=gen_nn_dir)
                        else:
                            evaluator = NNEval(gen_nn_dir)
                        eval_results = evaluator.evaluate(code_file)
                        eval_info = {str(evaluator.get_args()): eval_results}
                        name, new_accuracy, new_accuracy_to_time, code_quality = eval_results

                        with open(gen_nn_dir / 'eval_info.json', 'w+') as f:
                            json.dump(eval_info, f)
                        if nn_accepted(gen_nn_dir):
                            copy_to_lemur(df, gen_nn_dir, name)
                            break
                    except Exception as error:
                        print('failed to determine accuracy for', cv_model)
                        create_file(gen_nn_dir, f'error_{tries}.txt', str(traceback.format_exc()))  # Track traceback to have rich information on the exception.
                        with open(code_file, 'r') as f:
                            code_txt = f.read()
                        release_memory()
                        if not nn_regenerate_after_exception:
                            break
                        try:
                            new_code, new_hp, full_out = chat_bot.chat(
                                'The error "' + str(error) +
                                '" was occurred in the following code. fix this problem. '
                                "Provide only the code. Don't provide any explanation. Remove any text from this reply. + \n " +
                                code_txt,
                                engineer_prompt=False,
                                max_new_tokens=max_new_tokens)
                            print(f'2 gen hyperparams: {new_hp}')
                            create_file(gen_nn_dir, hp_file, new_hp)
                            create_file(gen_nn_dir, new_nn_file, new_code)
                            create_file(gen_nn_dir, new_out_file, full_out)
                        except:
                            pass
                        release_memory()
    print('[DEBUG] Release_memory.')
    release_memory()
    print('Clear LEMUR query cache.')
    lemur.data.cache_clear()
    print('The cache has been cleared.')
