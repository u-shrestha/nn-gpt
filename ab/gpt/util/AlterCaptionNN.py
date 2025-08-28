import json
import os
import shutil
from pathlib import Path
from typing import Optional

import ab.nn.api as nn_dataset
from tqdm import tqdm

from ab.nn.util.Util import create_file
from ab.gpt.util.Const import (
    conf_test_dir,
    epoch_dir,
    synth_dir,
    new_nn_file,
    new_out_file,
)
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code


def alter(
    epochs: int,
    test_conf: str,
    llm_name: str,
    gguf_file: Optional[str] = None,
    only_nn: Optional[str] = None,
) -> None:
    """
    Generate altered captioning models.

    Args:
        epochs: number of generation epochs (A0..A{epochs-1})
        test_conf: filename (under conf_test_dir) for the prompt JSON
        llm_name: HF model id or local path
        gguf_file: optional GGUF file for local/llama.cpp backends
        only_nn: if set, generate variants ONLY for this NN name (e.g. 'RESNETLSTM')
    """
    # 1) Load prompt configuration
    with open(conf_test_dir / test_conf, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    # 2) Load LLM + tokenizer
    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop...")

    # 3) Reset epoch output root
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    # 4) Epoch loop
    for ep in range(epochs):
        base_out = epoch_dir(ep)

        prompts = []  # (the_prompt:str, caption_row:pd.Series, addons_df:Optional[pd.DataFrame])

        # Build prompts for every template key
        for key in prompt_dict.keys():
            # Join prompt lines into a single template string
            prompt_tpl = "\n".join(prompt_dict[key]["prompt"]) + "\n"

            # Pull captioning rows (1 sample per NN)
            cap_df = (
                nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["task"])
                .groupby(by="nn")
                .sample(n=1)
            )

            # Optional filter to a single NN by name
            if only_nn:
                cap_df = cap_df[cap_df["nn"] == only_nn]
                if len(cap_df) == 0:
                    print(
                        f"[WARN] No rows found for nn='{only_nn}' under task='{prompt_dict[key]['task']}'. Skipping."
                    )
                    continue

            # Pull classification addons if configured
            addon_df = None
            if prompt_dict[key].get("addon_task"):
                addon_df = nn_dataset.data(task=prompt_dict[key]["addon_task"])

            # For each chosen captioning row, fill placeholders
            addon_entries = prompt_dict[key].get("addon_list", [])  # list of dicts
            K = len(addon_entries)

            for _, cap_row in cap_df.iterrows():
                # Base placeholder dict from captioning row
                para_dict = {}
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it["para"]] = cap_row[it["value"]]

                # Choose up to K distinct classification inspiration rows
                chosen_addons = None
                if addon_df is not None and len(addon_df) > 0 and K > 0:
                    filtered = addon_df.loc[addon_df.nn != cap_row["nn"]] if len(addon_df) > 0 else addon_df
                    base_pool = filtered if len(filtered) > 0 else addon_df
                    n_pick = min(K, len(base_pool))
                    chosen_addons = base_pool.sample(n=n_pick, replace=False, random_state=ep)
                    for i, entry in enumerate(addon_entries):
                        if i < chosen_addons.shape[0]:
                            para_dict[entry["para"]] = chosen_addons.iloc[i][entry["value"]]
                        else:
                            para_dict[entry["para"]] = ""
                else:
                    for entry in addon_entries:
                        para_dict[entry["para"]] = ""

                # Render final prompt text
                the_prompt = prompt_tpl.format(**para_dict)
                prompts.append((the_prompt, cap_row, chosen_addons))

        # 5) Generate code for each prepared prompt
        B_index = 0
        for idx, (the_prompt, cap_row, addons_df) in tqdm(
            enumerate(prompts), total=len(prompts), desc="Generate Codes"
        ):
            model_dir = synth_dir(base_out) / f"B{B_index}"
            code_file = model_dir / new_nn_file           # generated: new_nn.py
            df_file = model_dir / "dataframe.df"          # original captioning row
            orig_caption_py = model_dir / "original_caption.py"

            # --- Tokenize with chat template
            messages = [
                {"role": "system", "content": CODE_ONLY_SYSTEM},
                {"role": "user", "content": the_prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id  # safe fallback

            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            # --- Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=32 * 1024,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.08,
                num_beams=1,   
                num_return_sequences=1,
            )

            # --- Decode only new tokens (strip prompt echo)
            prompt_len = int(input_ids.shape[-1])
            gen_tokens = outputs[0][prompt_len:]
            out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            print("Response Available!")
            nn_code = extract_code(out_text)

            # Always create dir and save raw response + original artifacts
            model_dir.mkdir(exist_ok=True, parents=True)
            create_file(model_dir, new_out_file, out_text)  # raw LLM response

            # Save original captioning row (pickle) and its code
            try:
                cap_row.to_pickle(df_file)
            except Exception as e:
                print(f"[WARN] Failed to pickle caption row: {e}")
            try:
                with open(orig_caption_py, "w", encoding="utf-8") as f:
                    f.write(str(cap_row.get("nn_code", "")))
            except Exception as e:
                print(f"[WARN] Failed to save original caption code: {e}")

            # Save chosen add-ons' code as plain numbered files
            if addons_df is not None and len(addons_df) > 0:
                addons_dir = model_dir / "addons"
                addons_dir.mkdir(exist_ok=True, parents=True)
                for j, (_, addon_row) in enumerate(addons_df.iterrows(), start=1):
                    addon_file = addons_dir / f"addon_{j}.py"
                    try:
                        with open(addon_file, "w", encoding="utf-8") as f:
                            f.write(str(addon_row.get("nn_code", "")))
                    except Exception as e:
                        print(f"[WARN] Failed to save add-on #{j}: {e}")

            # Save generated code if present
            if nn_code:
                print(f"[INFO]Saving code to: {code_file}")
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(nn_code)
                B_index += 1
            else:
                print("[INFO]Response Invalid! (no fenced code block found)")
                # keep artifacts for debugging and continue
                continue
