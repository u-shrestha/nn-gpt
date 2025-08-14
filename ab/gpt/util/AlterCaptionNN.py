import json
import os
import shutil
import re
from pathlib import Path

import torch

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code

# --- Export helper (kept minimal and self-contained) --------------------------

def _export_single_file(src_file: Path, original_nn_name: str, final_dir: Path) -> Path:
    """
    Copy src_file to final_dir as <original_nn_name>_ALTER_<N>.py (N auto-incremented).
    Safe to call repeatedly; it scans existing files to pick the next number.
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    # strip .py if user stored names like 'ResNetTransformer.py'
    base = original_nn_name.rsplit('.', 1)[0]
    pat = re.compile(re.escape(base) + r"_ALTER_(\d+)\.py$")

    max_n = 0
    for p in final_dir.glob(f"{base}_ALTER_*.py"):
        m = pat.search(p.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass

    out_path = final_dir / f"{base}_ALTER_{max_n + 1}.py"
    shutil.copy2(src_file, out_path)
    print(f"[INFO]Exported final single-file model -> {out_path}")
    return out_path

# ------------------------------------------------------------------------------

def alter(epochs, test_conf, llm_name, gguf_file=None, final_out_dir=None, only_nn=None):
    """
    Args (new):
        only_nn (str|None): if set, generate variants ONLY for this NN name
                            (e.g. 'RESNETLSTM', 'ResNetTransformer').
    """
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

        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]['prompt']:
                prompt += pr + "\n"

            # pull captioning rows (1 per nn)
            data = nn_dataset.data(
                only_best_accuracy=True,
                task=prompt_dict[key]['task']
            ).groupby(by="nn").sample(n=1)

            # NEW: filter to a single NN if requested
            if only_nn:
                data = data[data["nn"] == only_nn]
                if len(data) == 0:
                    print(f"[WARN] No rows found for nn='{only_nn}' under task='{prompt_dict[key]['task']}'. Skipping.")
                    continue

            # addon rows (optional)
            addon_data = nn_dataset.data(
                only_best_accuracy=True,
                task=prompt_dict[key]['addon_task']
            ) if prompt_dict[key].get('addon_task') else None

            for _, row in data.iterrows():
                para_dict = {}
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]

                if addon_data is not None and len(addon_data) > 0:
                    filtered = addon_data.loc[addon_data.nn != row['nn']]
                    addon_row = (filtered if len(filtered) > 0 else addon_data).sample(n=1).iloc[0]
                    for it in prompt_dict[key].get('addon_list', []):
                        para_dict[it['para']] = addon_row[it['value']]

                prompts.append((prompt.format(**para_dict), row))

        # Produce new CV models
        B_index = 0
        for idx, item in tqdm(enumerate(prompts), desc="Generate Codes"):
            the_prompt, origdf = item
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'

            # Chat template -> generate
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': the_prompt}],
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                inputs,
                max_new_tokens=10000,
                do_sample=True,
                temperature=0.6,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )

            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("Response Available!")
            nn_code = extract_code(out)

            if nn_code:
                # Save to synth dir (existing behaviour)
                print(f"[INFO]Saving code to: {code_file}")
                code_file.parent.mkdir(exist_ok=True, parents=True)
                with open(code_file, 'w') as file:
                    file.write(nn_code)
                create_file(model_dir, new_out_file, out)

                # Save metadata/df
                if origdf is None:
                    if os.path.isfile(df_file):
                        os.remove(df_file)
                else:
                    orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                    with open(orig_code_file, 'w') as file:
                        file.write(origdf['nn_code'])
                    origdf.to_pickle(df_file)

                # NEW: optional export to final_out_dir
                if final_out_dir:
                    try:
                        # Prefer 'nn' column; fallback to a generic name
                        orig_name = str(origdf.get('nn', 'CAPTIONNET'))
                        _export_single_file(code_file, orig_name, Path(final_out_dir))
                    except Exception as e:
                        # Never crash the main pipeline; just log
                        print(f"[WARN] Final export skipped due to error: {e}")

                B_index += 1
            else:
                print("[INFO]Response Invalid!")
                continue