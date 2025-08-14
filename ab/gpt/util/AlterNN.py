import json
import os
import shutil

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code


def alter(epochs, test_conf, llm_name, gguf_file=None):
    # Load test prompts
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop...")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # Generate Prompts
        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]['prompt']:
                prompt += pr + "\n"
            # Get nn-dataset codes
            data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
            # Get addon nn-dataset codes
            addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]
                if not (addon_data is None):
                    ## Avoid sampling the same nn_code
                    addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
                    for it in prompt_dict[key]['addon_list']:
                        para_dict[it['para']] = addon_row[it['value']]
                prompts.append((prompt.format(**para_dict), row))

        # produce new CV models
        B_index = 0
        for idx, prompt in tqdm(enumerate(prompts), desc="Generate Codes"):
            prompt, origdf = prompt
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, ], add_generation_prompt=True, return_tensors="pt").to(model.device)
            # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
            outputs = model.generate(inputs, max_new_tokens=10000, do_sample=True, temperature=0.6, top_k=50, top_p=0.95, num_return_sequences=1,
                                     eos_token_id=tokenizer.eos_token_id)
            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("Response Available!")
            nn_code = extract_code(out)
            if nn_code:
                print(f"[INFO]Saving code to: {code_file}")
                code_file.parent.mkdir(exist_ok=True, parents=True)  # Move here to avoid empty folder
                with open(code_file, 'w') as file:
                    file.write(nn_code)
                create_file(model_dir, new_out_file, out)
                if origdf is None:
                    if os.path.isfile(df_file):  # Clean up dataframe.df, if no additional information generated this time.
                        os.remove(df_file)
                else:
                    # Store DataFrame information, mainly for passing parameters to evaluator.
                    orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                    with open(orig_code_file, 'w') as file:
                        file.write(origdf['nn_code'])

                    origdf.to_pickle(df_file)
                B_index += 1
            else:
                print("[INFO]Response Invalid!")
                continue 