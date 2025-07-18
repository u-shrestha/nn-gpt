import os
import sys
import tempfile
import importlib
import json
import ast
import torch
from glob import glob

from ab.gpt.util.LLM import LLM

# --- Code evaluation and training utilities ---
from ab.nn.util.CodeEval import evaluate_single_file
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train

from transformers import BitsAndBytesConfig, AutoTokenizer
from ab.nn.util.Const import out_dir

# --- Path setup (edit as needed) ---
NN_DATASET_DIR = "/home/krunaljesani/fork/nn-dataset/ab/nn/nn"
NN_STAT_DIR = "/home/krunaljesani/fork/nn-dataset/ab/nn/stat"
CAPTION_PROMPT_FILE = "/home/krunaljesani/fork/nn-gpt/ab/gpt/conf/prompt/test/NN_merge_caption.json"

# Tokenizer for prompt filling (not used by LLM unless you need it)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
llm = LLM(
    model_path="deepseek-ai/deepseek-coder-1.3b-instruct",
    bnb_config=bnb_config,
    base_path=out_dir
)

def get_stat_folder_for_model(model_filename):
    """Finds the stat folder for a given model, e.g. img-captioning_coco_bleu_RESNETLSTM."""
    model_basename = os.path.splitext(os.path.basename(model_filename))[0]
    pattern = os.path.join(NN_STAT_DIR, f"img-captioning_*_*_{model_basename}")
    folders = glob(pattern)
    if folders:
        return folders[0]  # Use the first match
    return None

def get_bleu_from_stat(model_filename):
    stat_folder = get_stat_folder_for_model(model_filename)
    if not stat_folder:
        return 0.0, 0, {}
    # Pick the latest (highest number) JSON file, or just '1.json' for now
    stat_files = glob(os.path.join(stat_folder, "*.json"))
    if not stat_files:
        return 0.0, 0, {}
    stat_files.sort()
    stat_file = stat_files[-1]  # Use the last (probably latest)
    try:
        with open(stat_file) as f:
            stats = json.load(f)
        bleu = float(stats.get("bleu", stats.get("accuracy", 0.0)))
        epoch = int(stats.get("epoch", 1)) if "epoch" in stats else int(os.path.splitext(os.path.basename(stat_file))[0])
        return bleu, epoch, stats
    except Exception:
        return 0.0, 0, {}

def list_captioning_models():
    """List only .py models that have img-captioning stat folders."""
    pattern = os.path.join(NN_DATASET_DIR, "*.py")
    all_models = [f for f in glob(pattern) if os.path.isfile(f)]
    captioning_models = []
    for f in all_models:
        if get_stat_folder_for_model(f):  # Only add if stat folder exists
            captioning_models.append(f)
    return captioning_models

def extract_class_source(code, class_name):
    """Extract class definition from code by name."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            lines = code.splitlines()
            start = node.lineno - 1
            end = start + 1
            class_indent = len(lines[start]) - len(lines[start].lstrip())
            while end < len(lines):
                if (len(lines[end].lstrip()) > 0 and
                    len(lines[end]) - len(lines[end].lstrip()) <= class_indent and
                    not lines[end].lstrip().startswith("#")):
                    break
                end += 1
            return "\n".join(lines[start:end])
    return None

def extract_xml_blocks(response: str, tag: str) -> str:
    import re
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_unique_model_name(base_name):
    import uuid
    return f"{base_name}_gpt_{str(uuid.uuid4())[:8]}"

def save_new_model_code(new_code, new_name):
    new_path = os.path.join(NN_DATASET_DIR, f"{new_name}.py")
    with open(new_path, "w") as f:
        f.write(new_code)
    print(f"[INFO] Saved improved model as {new_path}")
    return new_path

def build_single_prompt(prompts_path, context, prompt_key="merge_improve_captioning"):
    """
    Loads prompt template from prompts_path, fills with context, and returns string.
    """
    with open(prompts_path) as f:
        prompt_dict = json.load(f)
    prompt_block = prompt_dict[prompt_key]['prompt']
    # The prompt is a list of lines, join as string
    prompt = "\n".join(prompt_block)
    # Fill in context values using format
    prompt_filled = prompt.format(**context)
    return prompt_filled

def call_llm(llm, prompt, max_new_tokens=2048, temperature=0.2):
    tokenizer = llm.get_tokenizer()
    model = llm.get_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Or True if you want sampling
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    # Optionally, cut result after prompt if model echos prompt
    if result.startswith(prompt):
        result = result[len(prompt):].strip()
    return result


def main_merge_and_improve(train_epochs=8):
    print("=== Discovering and ranking candidate models ===")
    model_files = list_captioning_models()
    print("Captioning models selected:", [os.path.basename(f) for f in model_files])
    model_candidates = []
    for f in model_files:
        code = open(f).read()
        bleu, epoch, stats = get_bleu_from_stat(f)
        model_candidates.append((bleu, epoch, code, f))
    # Sort by BLEU (or accuracy)
    model_candidates.sort(reverse=True, key=lambda x: x[0])
    top_candidates = model_candidates[:4]
    print("Top candidate models (by BLEU or accuracy):")
    for bleu, epoch, _, fname in top_candidates:
        print(f"  {os.path.basename(fname)}: BLEU/Acc={bleu}, epoch={epoch}")

    # Extract Net class codes from top models
    top_codes = [extract_class_source(code, "Net") or code for _, _, code, _ in top_candidates]
    top_bleus = [bleu for bleu, _, _, _ in top_candidates]
    top_epochs = [epoch for _, epoch, _, _ in top_candidates]
    # Pad to length 4 if fewer models
    while len(top_codes) < 4: top_codes.append("")
    while len(top_bleus) < 4: top_bleus.append(0.0)
    while len(top_epochs) < 4: top_epochs.append(0)

    # Prepare context for prompt
    context = {
        "top_code_1": top_codes[0], "top_bleu_1": top_bleus[0], "top_epoch_1": top_epochs[0],
        "top_code_2": top_codes[1], "top_bleu_2": top_bleus[1], "top_epoch_2": top_epochs[1],
        "top_code_3": top_codes[2], "top_bleu_3": top_bleus[2], "top_epoch_3": top_epochs[2],
        "top_code_4": top_codes[3], "top_bleu_4": top_bleus[3], "top_epoch_4": top_epochs[3]
    }

    print("\n=== Sending composite prompt to LLM ===\n")
    prompt = build_single_prompt(CAPTION_PROMPT_FILE, context)
    #print(prompt)
    response = call_llm(llm, prompt)
    print("\n=== RAW LLM Output ===\n", repr(response))
    print("\n=== LLM Response ===\n")
    #print(response)

    new_prm_json = extract_xml_blocks(response, "hp")
    if not new_prm_json:
        print("[ERROR] No <hp>...</hp> block found in LLM output!")
    new_nn_code = extract_xml_blocks(response, "nn")
    if not new_nn_code:
        print("[ERROR] No <nn>...</nn> block found in LLM output!")

    print("\n=== Parsed new hyperparameters ===\n", new_prm_json)
    print("\n=== Parsed new model code ===\n", new_nn_code)

    try:
        new_prm = json.loads(new_prm_json)
    except Exception as e:
        print("[ERROR] Failed to parse hyperparameters JSON:", e)
        new_prm = {}

    # Validate code
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(new_nn_code)
        tmp.flush()
        tmp_name = tmp.name

    eval_result = evaluate_single_file(tmp_name)
    if not eval_result.get("passed", False):
        print("[ERROR] LLM-generated code did not pass validation. Details:", eval_result)
        os.unlink(tmp_name)
        return None

    # Import Net class
    module_name = get_unique_model_name("Net")
    spec = importlib.util.spec_from_file_location(module_name, tmp_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    NetClass = getattr(mod, "Net")
    print("[INFO] Imported improved Net class.")

    # Use dataset/stats from the best (first) candidate
    dataset = "coco"
    out_shape, minimum_bleu, train_set, test_set = load_dataset(
        "image-captioning", dataset, new_prm.get("transform", None)
    )
    print(f"[INFO] Loaded dataset: train={len(train_set)}, test={len(test_set)}")
    trainer = Train(
        config=("image-captioning", dataset, "bleu", new_nn_code),
        out_shape=out_shape,
        minimum_accuracy=minimum_bleu,
        batch=new_prm.get("batch", 64),
        model_name=module_name,
        task="image-captioning",
        train_dataset=train_set,
        test_dataset=test_set,
        metric="bleu",
        num_workers=new_prm.get("num_workers", 1),
        prm=new_prm,
        save_to_db=True,
        is_code=True,
        save_path=None
    )
    result, duration = trainer.train_n_eval(train_epochs)
    print(f"[RESULT] BLEU/Acc={result}, duration={duration}")

    # Save improved model
    new_name = get_unique_model_name("Net")
    save_new_model_code(new_nn_code, new_name)
    os.unlink(tmp_name)

    return result, new_nn_code, new_prm

if __name__ == "__main__":
    main_merge_and_improve(train_epochs=5)
