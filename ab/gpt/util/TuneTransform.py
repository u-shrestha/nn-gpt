import os
import shutil
from os import makedirs
from os.path import isfile
import glob
import pandas as pd

# import ab.nn.api as lemur 
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from peft import (PeftModel)
from tqdm import tqdm
import json


from ab.gpt.brute.trans.TransformEvalFt import run_evaluations
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import *
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.LoRA import LoRA
from ab.gpt.util.Util import exists
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders


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
    # flatten batched list
    all_chunks = sum(data["chunks"], []) 
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


def tune(test_nn, nn_train_epochs, skip_epoch, llm_path, llm_tune_conf, nn_gen_conf, conf_keys, llm_conf, training_args, peft_config,
         max_prompts=None, save_llm_output=True, max_new_tokens=16 * 1024, nn_name_prefix=None, temperature=1.0, top_k=50, top_p=0.9, test_metric=None):
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

    # Load generation prompt config
    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
        model = model.merge_and_unload()

    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config,
        test_metric=test_metric)

    print('Using Max Length:', model_loader.get_max_length())

    chat_bot = ChatBot(model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped transform generation at epoch {epoch}')
        else:
            # Pass folder paths to nn_gen
            nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
        
        print(f'[DEBUG]Perform finetune at epoch {epoch}.')
        
        data_processor = TransformGenPrompt(
            context_length if context_length else model_loader.get_max_length(), 
            tokenizer, 
            train_config_path
        )

        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts)
        
        print('Dataset length:', len(dataset))
        if len(dataset) > 0:
            model.train()
            model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
        else:
            print("Skipping training for epoch {epoch}: No data in dataset.")
            
        del dataset
        release_memory()


def nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    print('Preparing prompts for generation, this might take a while...')
    
    
    out_gen_dir = str(trans_dir / 'epoch1')
    result_gen_dir = str(trans_dir / 'result-e1')
  
    prompts = []
    
    # Load all data from folders to be used for seed prompts
    all_data = load_data_from_folders(out_gen_dir, result_gen_dir, only_best_accuracy=True)
    if len(all_data) == 0:
        print("Warning: No data loaded from folders for generation. Skipping.", flush=True)
        return
        
    for key in conf_keys:
        prompt_config = prompt_dict_global[key]
        prompt = ''
        for pr in prompt_config['prompt']:
            prompt += pr + '\n'
            
        # Get seed data
        if len(all_data) < test_nn:
            print(f"Warning: Requested {test_nn} samples, but only {len(all_data)} available. Using all.", flush=True)
            data_sample = all_data.sample(n=len(all_data))
        else:
            data_sample = all_data.sample(n=test_nn)

        addon_data = all_data
        
        for _, row in data_sample.iterrows():
            para_dict = dict()
            row_dict = row.to_dict()
            for it in prompt_config['input_list']:
                para_dict[it['para']] = row_dict.get(it['value'])
            
            # Avoid sampling the same transform
            filtered_addon_data = addon_data.loc[addon_data.id_name != row['id_name']]
            if len(filtered_addon_data) > 0:
                addon_row = filtered_addon_data.sample(n=1).iloc[0].to_dict()
                if prompt_config.get('addon_list'):
                    for it in prompt_config['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])
                prompts.append((prompt.format(**para_dict), row))
            else:
                print(f"Warning: Could not find addon data for {row['id_name']}. Skipping prompt.", flush=True)
                
    models_dir = synth_dir(out_path)
    
    for idx, prompt_data in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt, origdf = prompt_data
        
        
        code, hp, _, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)
        
        if save_llm_output: create_file(model_dir, new_out_file, full_out)
        makedirs(model_dir, exist_ok=True)

      
        tr_content = None
        
        # Try finding standard <tr> tags
        if '<tr>' in full_out:
            start = full_out.find('<tr>') + 4
            end = full_out.find('</tr>', start)
            if end != -1:
                tr_content = full_out[start:end]
        
        # Look for the python code block if <tr> fails
        if not tr_content and 'def transform' in full_out:
            start = full_out.find('def transform')
            tr_content = full_out[start:] 
            if '</' in tr_content:
                tr_content = tr_content.split('</')[0]
        
        # Clean and Save
        if tr_content:
            tr_content = tr_content.replace('```python', '').replace('```', '').strip()
            try:
                print(f'Generated transformer:\n\n{tr_content}\n----\n')
                create_file(model_dir, transformer_file, tr_content)
            except Exception as e:
                print(f"Error saving transformer: {e}")
                continue 
        else:
            print(f"Warning: Could not extract valid transform code for B{idx}. Skipping.")
            continue # Skip to next prompt if no code found
      
        # Save DataFrame info
        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):
                os.remove(df_file)
        else:
            create_file(model_dir, f"original_{origdf['id_name']}.py", origdf['transform_code'])
            origdf.to_pickle(df_file)
            
    print('[DEBUG] Release memory.')
    release_memory()
    
    # Evaluate produced CV models
    if exists(models_dir):
        try:
            run_evaluations(epoch)
        except Exception as e:
            print(f"Error running evaluation main(): {e}", flush=True)
            
        print('[DEBUG] Release_memory.')
        release_memory()
        
    print('Folder data reload will occur next epoch.')