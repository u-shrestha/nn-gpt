import os
import json
import random
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from ab.nn.api import data  # Import your actual data loader

from ab.gpt.util.Const import conf_train_dir, conf_llm_dir

# Step 1: Load model metadata
df = data()
df["prm_str"] = df["prm"].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x))

# Step 2: Drop duplicates based on critical fields including `prm_str`
unique_cols = ['task', 'nn', 'transform_code', 'metric', 'metric_code']
df_unique = df.drop_duplicates(subset=unique_cols).reset_index(drop=True)

# Step 3: Load prompt templates and LLM model config

with open(conf_train_dir / 'NN_Layers.json', "r") as f:
    templates = json.load(f)
with open(conf_llm_dir / 'ds_coder_1.3b_instruct.json', "r") as f:
    model_config = json.load(f)

model_name = model_config["base_model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Step 4: Select best classification model as backbone for segmentation (optional)
best_class_code = None
if not df_unique[df_unique["task"] == "img-classification"].empty:
    class_subset = df_unique[df_unique["task"] == "img-classification"].copy()
    class_subset["accuracy"] = pd.to_numeric(class_subset["accuracy"], errors="coerce")
    best_row = class_subset.loc[class_subset["accuracy"].idxmax()]
    best_class_code = str(best_row["nn_code"])

df_unique["nn_code_generated"] = None

# Step 5: Iterate and generate
for idx, row in df_unique.iterrows():
    task = row["task"]
    original_code = str(row["nn_code"])
    prm_str = row["prm_str"]
    prompt = ""
    addon_accuracy_val = None
    template_key = ""

    if task == "img-classification":
        template_key = "improvement_classification_codeonly"
        template = templates[template_key]

        try:
            acc_val = float(row["accuracy"])
        except Exception:
            acc_val = 0.5

        base = acc_val if acc_val <= 1 else acc_val / 100.0
        addon_accuracy_val = round(min(base + random.uniform(0.01, 0.05), 0.9999), 4)
        addon_accuracy_fmt = addon_accuracy_val if acc_val <= 1 else round(addon_accuracy_val * 100, 2)

        accuracy_val = row["accuracy"]
        prompt_lines = [line.format(
            nn_code=original_code,
            accuracy=accuracy_val,
            addon_accuracy=addon_accuracy_fmt
        ) for line in template["prompts"]]
        prompt = "\n".join(prompt_lines)

    elif task == "img-segmentation" and best_class_code:
        template_key = "improvement_segmentation_codeonly"
        template = templates[template_key]
        prompt_lines = [line.format(
            nn_code=original_code,
            addon_nn_code=best_class_code
        ) for line in template["prompts"]]
        prompt = "\n".join(prompt_lines)

    else:
        print(f"[Skipped] Task '{task}' at row {idx} not supported.")
        continue

    # Step 6: LLM generation
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    outputs = model.generate(input_ids, max_new_tokens=2046, do_sample=False)
    gen_tokens = outputs[0][input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    gen_code = gen_text.strip()

    df_unique.at[idx, "nn_code_generated"] = gen_code

    # Step 7: Save artifacts
    task_dir = os.path.join("output", task)
    sanitized_prm = "".join(c for c in prm_str if c.isalnum() or c in ['_', '-'])[:50]
    model_folder = f"{row['nn']}_{sanitized_prm}"
    output_dir = os.path.join(task_dir, model_folder)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "original.py"), "w") as f:
        f.write(original_code)
    with open(os.path.join(output_dir, "generated.py"), "w") as f:
        f.write(gen_code)
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    metadata = {
        "task": row["task"],
        "nn": row["nn"],
        "prm": row["prm"],
        "transform_code": row["transform_code"],
        "accuracy": row["accuracy"],
        "addon_accuracy": addon_accuracy_fmt if task == "img-classification" else None,
        "metric": row["metric"],
        "metric_code": row["metric_code"],
        "duration": row["duration"],
        "template_used": template_key,
        "model_name": model_name
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[✓] Generated for {task} model '{row['nn']}' → saved in {output_dir}")

print("\n🎯 All models processed. Output stored with new column 'nn_code_generated' in DataFrame.")
