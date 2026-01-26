import json
import os
import shutil
import torch

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code, extract_delta


def format_prompt_with_supporting_models(prompt_template, para_dict, supporting_models):
    """
    Format prompt template with supporting models information.
    Handles complex formatting including supporting models display.
    """
    para_dict["n"] = len(supporting_models) if supporting_models else 0

    if supporting_models:
        supporting_models_text = ""
        for i, model in enumerate(supporting_models, 1):
            supporting_models_text += f"\nSupporting Model {i}:\n"
            for key, value in model.items():
                supporting_models_text += f"  {key}: {value}\n"
        para_dict["supporting_models_prompt"] = supporting_models_text
    else:
        para_dict["supporting_models_prompt"] = "No supporting models available."

    try:
        formatted_prompt = prompt_template.format(**para_dict)
    except KeyError as e:
        print(f"[WARNING] Missing parameter in prompt template: {e}")
        formatted_prompt = prompt_template
        for key, value in para_dict.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

    return formatted_prompt


def alter(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50, *args, **kwargs):
    """
    Generate improved neural network models using full-code generation.

    Features included:
    - Batched inference (kwargs['batch_size'], default 8)
    - GPU-safe tokenization (pad_token set to eos if missing)
    - Left padding for decoder-only models
    - Supporting models injection (n samples from addon_task)
    - inference_gpt_oss length guard (skip long prompts)
    - Always writes raw LLM outputs (new_out_file) for every attempt
    - Stable per-attempt B-index folder mapping (one folder per attempted generation)
    - Restores tokenizer.padding_side on exit
    """
    inference_gpt_oss = kwargs.get("inference_gpt_oss", False)
    inference_gpt_oss_max_input_length = kwargs.get("inference_gpt_oss_max_input_length", None)

    batch_size = int(kwargs.get("batch_size", 1))
    batch_size = max(1, batch_size)

    max_input_tokens = int(kwargs.get("max_input_tokens", 8192))
    max_new_tokens = int(kwargs.get("max_new_tokens", 8 * 1024))

    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    # --- GPU-safe padding setup ---
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    print(
        f"Load Model Complete. Batched generation enabled (batch_size={batch_size}). "
        f"Will fetch {n} supporting models per prompt."
    )

    def _prompt_len_ids(text: str) -> int:
        try:
            return len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            return len(tokenizer(text).input_ids)

    shutil.rmtree(epoch_dir(), ignore_errors=True)

    try:
        for epoch in range(epochs):
            out_path = epoch_dir(epoch)

            prompts = []
            for key in prompt_dict.keys():
                prompt = ""
                for pr in prompt_dict[key]["prompt"]:
                    prompt += pr + "\n"

                # Get nn-dataset codes
                data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["task"]).groupby(by="nn").sample(n=1)

                # Get addon nn-dataset codes
                addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["addon_task"])

                for _, row in data.iterrows():
                    para_dict = dict()
                    for it in prompt_dict[key]["input_list"]:
                        para_dict[it["para"]] = row[it["value"]]

                    # Fetch n supporting models
                    supporting_models = []
                    if not (addon_data is None) and n > 0:
                        available_addon_data = addon_data.loc[addon_data.nn != row["nn"]]
                        n_samples = min(n, len(available_addon_data))
                        if n_samples > 0:
                            addon_rows = available_addon_data.sample(n=n_samples)
                            for _, addon_row in addon_rows.iterrows():
                                model_info = {}
                                for it in prompt_dict[key]["addon_list"]:
                                    model_info[it["para"]] = addon_row[it["value"]]
                                supporting_models.append(model_info)

                        para_dict["supporting_models"] = supporting_models

                        # Backward compatibility: also expose first model fields
                        if supporting_models:
                            first_model = supporting_models[0]
                            for it in prompt_dict[key]["addon_list"]:
                                para_dict[it["para"]] = first_model[it["para"]]

                    formatted_prompt = format_prompt_with_supporting_models(prompt, para_dict, supporting_models)

                    # Apply chat template as TEXT (not tokenized yet)
                    chat_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": formatted_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    if inference_gpt_oss and inference_gpt_oss_max_input_length is not None:
                        if _prompt_len_ids(chat_text) > inference_gpt_oss_max_input_length:
                            continue

                    prompts.append((chat_text, row))

            B_global_index = 0
            print(f"Total models to generate: {len(prompts)}")

            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Epoch {epoch} (Batched Generate Codes)"):
                batch = prompts[i : i + batch_size]
                batch_texts = [p[0] for p in batch]
                batch_rows = [p[1] for p in batch]

                model_inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(model.device)

                with torch.no_grad():
                    model.eval()
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=0.95,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )

                input_len = model_inputs.input_ids.shape[1]
                decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

                for out_text, origdf in zip(decoded_outputs, batch_rows):
                    model_dir = synth_dir(out_path) / f"B{B_global_index}"
                    code_file = model_dir / new_nn_file
                    df_file = model_dir / "dataframe.df"

                    model_dir.mkdir(parents=True, exist_ok=True)
                    create_file(model_dir, new_out_file, out_text)

                    nn_code = extract_code(out_text)
                    if nn_code:
                        with open(code_file, "w") as f:
                            f.write(nn_code)

                        if origdf is None:
                            if os.path.isfile(df_file):
                                os.remove(df_file)
                        else:
                            orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                            with open(orig_code_file, "w") as f:
                                f.write(origdf["nn_code"])
                            origdf.to_pickle(df_file)

                    B_global_index += 1
    finally:
        tokenizer.padding_side = old_padding_side


def alter_delta(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50):
    """
    Generate improved neural network models using delta-based approach.

    Similar to alter() but:
    1. LLM generates code deltas (unified diffs) instead of full code
    2. Deltas are applied to baseline code to reconstruct improved code
    3. Requires delta-enabled config file (with use_delta: true)
    """
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    use_delta = False
    for key in prompt_dict.keys():
        key_config = prompt_dict[key]
        if isinstance(key_config, dict):
            use_delta = key_config.get("use_delta", False) or "delta" in str(key).lower()
            if use_delta:
                break

    if not use_delta:
        print("[WARNING] Config file does not have delta mode enabled. Falling back to regular alter().")
        return alter(epochs, test_conf, llm_name, gguf_file, n, temperature, top_k)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop... (Delta mode enabled)")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]["prompt"]:
                prompt += pr + "\n"

            data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["task"]).groupby(by="nn").sample(n=1)

            addon_data = None
            if prompt_dict[key].get("addon_task"):
                addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["addon_task"])

            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it["para"]] = row[it["value"]]

                if prompt_dict[key].get("addon_list") and addon_data is not None:
                    supporting_models = []
                    if n > 0:
                        available_addon_data = addon_data.loc[addon_data.nn != row["nn"]]
                        n_samples = min(n, len(available_addon_data))
                        if n_samples > 0:
                            addon_rows = available_addon_data.sample(n=n_samples)
                            for _, addon_row in addon_rows.iterrows():
                                model_info = {}
                                for it in prompt_dict[key]["addon_list"]:
                                    model_info[it["para"]] = addon_row[it["value"]]
                                supporting_models.append(model_info)

                            para_dict["supporting_models"] = supporting_models

                            if supporting_models:
                                first_model = supporting_models[0]
                                for it in prompt_dict[key]["addon_list"]:
                                    para_dict[it["para"]] = first_model[it["para"]]

                    formatted_prompt = format_prompt_with_supporting_models(prompt, para_dict, supporting_models)
                    prompts.append((formatted_prompt, row))
                else:
                    prompts.append((prompt.format(**para_dict), row))

        B_index = 0
        for idx, prompt_item in tqdm(enumerate(prompts), desc="Generate Deltas"):
            prompt, origdf = prompt_item
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / "dataframe.df"

            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                inputs,
                max_new_tokens=64 * 1024,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            out = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
            print("Response Available!")

            delta = extract_delta(out)

            if delta and origdf is not None:
                try:
                    from ab.gpt.util.DeltaUtil import apply_delta, validate_delta

                    baseline_code = origdf.get("nn_code", "")

                    if baseline_code and validate_delta(delta):
                        nn_code = apply_delta(baseline_code, delta)
                        if nn_code:
                            code_file.parent.mkdir(exist_ok=True, parents=True)
                            with open(code_file, "w") as f:
                                f.write(nn_code)

                            create_file(model_dir, new_out_file, out)

                            orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                            with open(orig_code_file, "w") as f:
                                f.write(baseline_code)

                            delta_file = model_dir / "delta.diff"
                            with open(delta_file, "w") as f:
                                f.write(delta)

                            origdf.to_pickle(df_file)
                            B_index += 1
                            continue
                except Exception:
                    pass

            # Fallback to full code extraction
            nn_code = extract_code(out)
            if nn_code:
                code_file.parent.mkdir(exist_ok=True, parents=True)
                with open(code_file, "w") as f:
                    f.write(nn_code)

                create_file(model_dir, new_out_file, out)
                if origdf is not None:
                    origdf.to_pickle(df_file)
                B_index += 1
            else:
                continue
