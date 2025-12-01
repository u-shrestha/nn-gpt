"""
RAG-based SFT code generation utility.
Generates NNEval-compatible PyTorch models using finetuned LLMs with novelty filtering.
"""
import json, re, sys, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ab.gpt.util.Const import (
    conf_llm_dir,
    conf_test_dir,
    conf_prompt_dir,
    epoch_dir,
    synth_dir,
    new_nn_file,
)

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False

FENCE_PY = re.compile(r"```python\s*(.*?)```", re.S | re.M)
FENCE_ANY = re.compile(r"```\s*(.*?)```", re.S | re.M)
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")


def extract_python_block(s: str) -> str:
    """Extract largest fenced code block or return raw string."""
    m = FENCE_PY.findall(s)
    if not m:
        m = FENCE_ANY.findall(s)
    if not m:
        return s.strip()
    return max(m, key=len).strip()


def has_nn_module_subclass(src: str) -> bool:
    """Check if source defines nn.Module subclass."""
    try:
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    if getattr(b, "attr", "") == "Module" or getattr(b, "id", "") == "Module":
                        return True
        return False
    except Exception:
        return False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# Text novelty via MinHash-LSH
def _tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def _shingles(tokens: List[str], n: int = 7) -> List[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def to_minhash(code: str, num_perm: int = 128, n: int = 7):
    """Compute MinHash for code string."""
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(_tokenize(code), n=n):
        mh.update(sh.encode("utf-8"))
    return mh


def build_train_lsh(train_jsonl: Path, threshold: float = 0.85, num_perm: int = 128):
    """Build LSH index from training data for novelty detection."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    id2mh = {}
    rows = load_jsonl(train_jsonl)
    for i, row in enumerate(rows):
        msgs = row.get("messages", [])
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        code = extract_python_block(asst) if asst else ""
        if not code:
            continue
        mh = to_minhash(code, num_perm=num_perm)
        key = row.get("id", f"train_{i}")
        lsh.insert(key, mh)
        id2mh[key] = mh
    return lsh, id2mh


def nearest_jaccard(lsh, id2mh, mh) -> float:
    """Find maximum Jaccard similarity to indexed training samples."""
    cands = lsh.query(mh)
    best = 0.0
    for k in cands:
        j = mh.jaccard(id2mh[k])
        if j > best:
            best = j
    return best


# Code validation for NNEval compatibility
def validate_code_for_nneval(code: str) -> tuple:
    """Validate code structure for NNEval.py compatibility. Returns (is_valid, error_msg)."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (False, f"Syntax error: {e}")

    has_net = False
    net_methods = set()
    has_supported_hyperparams = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            has_net = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    net_methods.add(item.name)
        elif isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters":
            has_supported_hyperparams = True

    if not has_net:
        return (False, "No Net class")
    if not has_supported_hyperparams:
        return (False, "Missing supported_hyperparameters()")

    required = {"__init__", "forward", "train_setup", "learn"}
    missing = required - net_methods
    if missing:
        return (False, f"Missing methods: {missing}")

    if "import torch" not in code:
        return (False, "Missing 'import torch'")
    if "import torch.nn" not in code and "from torch import nn" not in code:
        return (False, "Missing torch.nn import")

    return (True, "")


# Model loading and prompt rendering
def ensure_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def load_model_and_tokenizer(
    base_model: str,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    ) if load_in_4bit else None

    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
    ensure_pad(tok)
    mdl = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=bnb_cfg,
    )
    return mdl, tok


def load_prompt_template(prompt_config_path: Path) -> Dict[str, str]:
    """Load prompt template from JSON config file."""
    try:
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load prompt template: {e}")
        return {
            "system_prompt_template": "",
            "prefix_code": "```python\nimport torch\nimport torch.nn as nn\n\n",
        }


def render_prompt(tokenizer, messages: List[Dict[str, str]], prompt_template: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Render prompt with enhanced system message for NNEval compatibility."""
    msgs = [m for m in messages if m["role"] in ("system", "user")]

    enhanced_msgs = []
    for msg in msgs:
        if msg["role"] == "system":
            enhanced_system = msg["content"] + "\n\n" + prompt_template.get("system_prompt_template", "")
            enhanced_msgs.append({"role": "system", "content": enhanced_system})
        else:
            enhanced_msgs.append(msg)

    text = tokenizer.apply_chat_template(enhanced_msgs, tokenize=False, add_generation_prompt=True)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return toks


# Known training blocks for rejection sampling
KNOWN_TRAINING_BLOCKS = {
    "DlaBasic",
    "InvertedResidual",
    "ConvBlock",
    "AirUnit",
    "AirInitBlock",
    "HardMish",
    "RMSNorm",
    "Distance",
    "ResBlock",
    "BasicBlock",
    "Bottleneck",
    "ConvNorm",
    "_InvertedResidual",
    "HardMishAutoFn",
}


def contains_known_blocks(code: str) -> bool:
    """Check if code contains known training block class names."""
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name in KNOWN_TRAINING_BLOCKS:
                    return True
                for known_block in KNOWN_TRAINING_BLOCKS:
                    if known_block.lower() in class_name.lower() and class_name != "Net":
                        return True
    except Exception:
        for known_block in KNOWN_TRAINING_BLOCKS:
            if f"class {known_block}" in code or f"class {known_block}(" in code:
                return True
    return False


@torch.inference_mode()
def generate_code(
    model, tokenizer, toks, device,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 50,
    prefix_code: Optional[str] = None,
) -> str:
    """Generate code from model with optional prefix for constrained decoding."""
    input_ids = toks["input_ids"].to(device)
    attn_mask = toks.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    if prefix_code:
        prefix_tokens = tokenizer(prefix_code, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([input_ids, prefix_tokens], dim=1)
        if attn_mask is not None:
            prefix_mask = torch.ones((attn_mask.shape[0], prefix_tokens.shape[1]), device=device, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, prefix_mask], dim=1)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    generated_tokens = gen[0][input_ids.shape[1]:]
    out = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if prefix_code:
        out = prefix_code + out
    return out


# Main generation function
def main(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epoch: int = 0,
    config: str = "nngpt_unique_arch_rag.json",
    base_model: Optional[str] = None,
    load_in_4bit: bool = True,
    device: Optional[str] = None,
    max_items: int = 300,
    max_new_tokens: int = 2048,
    temperature: float = 0.20,
    top_p: float = 0.9,
    top_k: int = 50,
    rejection_sampling: bool = True,
    max_rejections: int = 5,
    hf_token: Optional[str] = None,
    novelty: bool = True,
    near_dup_threshold: float = 0.85,
    train_data_dir: Optional[str] = None,
    save_failures: bool = True,
    force_valid_structure: bool = True,
    prompt_template: str = "unique_rag_test_rules.json",
):
    """
    Generate NNEval-compatible PyTorch models using a finetuned LLM.
    
    Args:
        data_dir: Folder with NN_Rag_gen_test.jsonl (default: conf_test_dir/NN_Rag_gen_test.jsonl)
        output_dir: Output directory for results (default: out/nngpt/llm/epoch/A{epoch}/)
        epoch: Epoch number for model directory structure (default: 0)
        config: JSON config file in conf/llm/ directory holding base_model_name
        base_model: Base model name (overrides config file if provided)
        load_in_4bit: Load model in 4-bit quantization (default: True)
        device: Device to run generation on (default: cuda if available else cpu)
        max_items: Max number of items from NN_Rag_gen_test.jsonl to process
        max_new_tokens: Max new tokens to generate per sample
        temperature: Generation temperature (default: 0.20)
        top_p: Nucleus sampling threshold (default: 0.9)
        top_k: Top-k sampling (default: 50)
        rejection_sampling: Enable rejection sampling (default: True)
        max_rejections: Max rejection attempts before accepting anyway (default: 5)
        hf_token: Hugging Face token if needed for model access
        novelty: Enable text MinHash/LSH novelty scoring & rejection (default: True)
        near_dup_threshold: NN-Jaccard >= threshold => near-duplicate (default: 0.85)
        train_data_dir: Folder with NN_Rag_gen_train.jsonl for novelty check
        save_failures: Save failure code blocks (default: True)
        force_valid_structure: Use prefix_code to enforce NNEval structure (default: True)
        prompt_template: Prompt template config file in conf/prompt/ (default: unique_rag_test_rules.json)
    """
    # Set device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompt template from config
    prompt_config_path = conf_prompt_dir / prompt_template
    prompt_tmpl = load_prompt_template(prompt_config_path)
    print(f"[INFO] Loaded prompt template from: {prompt_config_path}")

    # Set up model directory structure compatible with NNEval.py
    current_epoch_path = epoch_dir(epoch)
    models_base_dir = synth_dir(current_epoch_path)
    models_base_dir.mkdir(parents=True, exist_ok=True)

    # Output directory for results/summary
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = current_epoch_path

    # Output subdirs
    fail_dir = out_dir / "code_fail"
    fail_dir.mkdir(exist_ok=True)
    raw_output_dir = out_dir / "raw_outputs"
    raw_output_dir.mkdir(exist_ok=True)
    raw_fail_dir = out_dir / "raw_failures"
    raw_fail_dir.mkdir(exist_ok=True)

    print(f"[INFO] Models will be saved to: {models_base_dir}")
    print(f"[INFO] Results will be saved to: {out_dir}")
    print(f"[INFO]   - results.jsonl")
    print(f"[INFO]   - summary.json")

    # Load base model config if not explicitly given
    if base_model is None:
        config_path = conf_llm_dir / config
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_model = cfg.get("base_model_name")
                if base_model:
                    print(f"[INFO] Loaded base_model '{base_model}' from config file: {config_path}")
                else:
                    raise ValueError(f"Config file {config_path} does not contain 'base_model_name'")
            except Exception as e:
                print(f"[ERROR] Failed to load config from {config_path}: {e}")
                sys.exit(1)
        else:
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
    else:
        print(f"[INFO] Using explicitly provided base_model: {base_model}")

    # Load data
    if data_dir:
        test_path = Path(data_dir) / "NN_Rag_gen_test.jsonl"
    else:
        test_path = conf_test_dir / "NN_Rag_gen_test.jsonl"

    if not test_path.exists():
        print(f"[ERROR] NN_Rag_gen_test.jsonl not found at {test_path}")
        sys.exit(1)

    rows = load_jsonl(test_path)
    print(f"[INFO] Loaded {len(rows)} items from {test_path}")

    # Load model and tokenizer (base model only, no adapters)
    model, tokenizer = load_model_and_tokenizer(
        base_model,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
        hf_token=hf_token,
    )
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)
    try:
        if getattr(model.config, "use_cache", True):
            model.config.use_cache = False
    except Exception:
        pass

    # Novelty indices
    text_lsh = None
    text_id2mh = None
    if novelty:
        if not _HAS_DATASKETCH:
            print("[NOVELTY] datasketch not installed; disabling text near-dup.")
            novelty = False
        else:
            # Determine training data directory for novelty check
            if train_data_dir:
                tdir = Path(train_data_dir)
            elif data_dir:
                tdir = Path(data_dir)
            else:
                tdir = conf_prompt_dir / "train"  # Default: conf/prompt/train/

            tpath = tdir / "NN_Rag_gen_train.jsonl"
            if tpath.exists():
                print(f"[NOVELTY] indexing text MinHash on {tpath} (τ={near_dup_threshold}) ...")
                text_lsh, text_id2mh = build_train_lsh(tpath, threshold=near_dup_threshold)
            else:
                print(f"[NOVELTY] WARNING: {tpath} not found; text novelty disabled.")
                novelty = False

    # Iterate over prompts
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    done = ok = 0
    near_dup_hits = near_dup_total = 0

    with results_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(rows[: max_items]):
            mid = item.get("id", f"sample_{idx}")
            messages = item["messages"]
            user_txt = next((m["content"] for m in messages if m["role"] == "user"), "")

            toks = render_prompt(tokenizer, messages, prompt_tmpl)

            nn_j = None
            near_dup_flag = None
            raw = None
            rejection_count = 0
            current_temp = temperature

            try:
                # Rejection sampling loop:
                # 1) structure validation
                # 2) novelty rejection (highest priority after structure)
                # 3) known-block rejection
                while True:
                    prefix_code = (
                        prompt_tmpl.get("prefix_code", None)
                        if force_valid_structure
                        else None
                    )

                    raw = generate_code(
                        model,
                        tokenizer,
                        toks,
                        device_obj,
                        max_new_tokens=max_new_tokens,
                        temperature=current_temp,
                        top_p=top_p,
                        top_k=top_k,
                        prefix_code=prefix_code,
                    )

                    code = extract_python_block(raw)

                    # 1) Validate code structure
                    is_valid, error_msg = validate_code_for_nneval(code)
                    if not is_valid:
                        if rejection_sampling:
                            rejection_count += 1
                            if rejection_count >= max_rejections:
                                print(
                                    f"[WARN] {idx}: Reached max rejections "
                                    f"({max_rejections}), accepting despite validation error: {error_msg}"
                                )
                                # accept this one (structurally broken), no nn_j / near_dup
                                break
                            current_temp = min(1.0, current_temp + 0.1)
                            print(
                                f"[REJECT-STRUCT] {idx}: Attempt {rejection_count}/{max_rejections} "
                                f"- Validation failed: {error_msg}, retrying with temp={current_temp:.2f}"
                            )
                            continue
                        else:
                            # No rejection_sampling: accept but record the error in results
                            print(f"[WARN] {idx}: Invalid structure but rejection_sampling disabled: {error_msg}")
                            break

                    # 2) Novelty-based rejection (highest-priority non-structural filter)
                    candidate_nn_j = None
                    candidate_near_dup = None
                    if novelty and text_lsh is not None:
                        mh = to_minhash(code)
                        candidate_nn_j = nearest_jaccard(text_lsh, text_id2mh, mh)
                        candidate_near_dup = candidate_nn_j >= near_dup_threshold
                        if candidate_near_dup and rejection_sampling:
                            rejection_count += 1
                            if rejection_count >= max_rejections:
                                print(
                                    f"[WARN] {idx}: Reached max rejections "
                                    f"({max_rejections}), accepting despite near-duplicate "
                                    f"(J={candidate_nn_j:.4f} ≥ τ={near_dup_threshold})"
                                )
                                # accept this one (near-dup)
                                nn_j = candidate_nn_j
                                near_dup_flag = True
                                break
                            current_temp = min(1.0, current_temp + 0.1)
                            print(
                                f"[REJECT-NOVELTY] {idx}: Attempt {rejection_count}/{max_rejections} "
                                f"- J={candidate_nn_j:.4f} ≥ τ={near_dup_threshold}, retrying with temp={current_temp:.2f}"
                            )
                            continue

                    # 3) Known-block rejection (secondary to novelty)
                    if rejection_sampling and contains_known_blocks(code):
                        rejection_count += 1
                        if rejection_count >= max_rejections:
                            print(
                                f"[WARN] {idx}: Reached max rejections "
                                f"({max_rejections}), accepting despite known blocks"
                            )
                            # we still keep whatever novelty stats we computed above
                            nn_j = candidate_nn_j
                            near_dup_flag = candidate_near_dup
                            break
                        current_temp = min(1.0, current_temp + 0.1)
                        print(
                            f"[REJECT-BLOCK] {idx}: Attempt {rejection_count}/{max_rejections} "
                            f"- Known block detected, retrying with temp={current_temp:.2f}"
                        )
                        continue

                    # If we reached here, we accept this candidate
                    nn_j = candidate_nn_j
                    near_dup_flag = candidate_near_dup
                    break

                # Save raw output
                raw_output_path = raw_output_dir / f"gen_{idx:04d}_raw.txt"
                raw_output_path.write_text(raw, encoding="utf-8")

                if not has_nn_module_subclass(code):
                    raise RuntimeError("No nn.Module subclass in code block.")

                # Track novelty stats only for accepted code when novelty is enabled
                if novelty and text_lsh is not None and nn_j is not None:
                    near_dup_total += 1
                    if near_dup_flag:
                        near_dup_hits += 1

                # Save code in NNEval-compatible structure:
                #   out/nngpt/llm/epoch/A{epoch}/synth_nn/B{idx}/new_nn.py
                model_dir = models_base_dir / f"B{idx}"
                model_dir.mkdir(parents=True, exist_ok=True)
                code_path = model_dir / new_nn_file
                code_path.write_text(code, encoding="utf-8")

                rec = {
                    "id": mid,
                    "file": str(code_path),
                    "raw_output": str(raw_output_path),
                    "ok": True,
                    "nn_jaccard": round(nn_j, 4) if nn_j is not None else None,
                    "near_dup_text": bool(near_dup_flag) if near_dup_flag is not None else None,
                    "rejection_count": rejection_count if rejection_sampling else None,
                }
                ok += 1
            except Exception as e:
                # Save raw output for failures
                raw_fail_path: Optional[Path] = None
                if raw is not None:
                    raw_fail_path = raw_fail_dir / f"gen_{idx:04d}_raw.txt"
                    raw_fail_path.write_text(raw, encoding="utf-8")

                # Save code for failures (if requested)
                fail_code_path: Optional[Path] = None
                if save_failures:
                    fail_code_path = fail_dir / f"gen_{idx:04d}_fail.py"
                    try:
                        if "code" in locals() and code:
                            fail_code_path.write_text(code, encoding="utf-8")
                        elif raw is not None:
                            fail_code_path.write_text(
                                f"# Raw LLM output (code extraction failed):\n"
                                f"# {type(e).__name__}: {e}\n\n{raw}",
                                encoding="utf-8",
                            )
                        else:
                            fail_code_path.write_text(
                                f"# No code generated\n# Error: {type(e).__name__}: {e}",
                                encoding="utf-8",
                            )
                    except Exception as write_err:
                        print(f"[WARN] Failed to save failure code for {idx}: {write_err}")
                        fail_code_path = None

                rec = {
                    "id": mid,
                    "file": None,
                    "raw_output": str(raw_fail_path) if raw_fail_path is not None else None,
                    "fail_code": str(fail_code_path) if fail_code_path is not None else None,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "nn_jaccard": None,
                    "near_dup_text": None,
                }

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done += 1
            if done % 10 == 0:
                print(f"[{done}/{min(max_items, len(rows))}] ok={ok}")

    # Summary
    summary = {
        "counts": {"done": done, "ok": ok, "fail": done - ok},
        "rates": {
            "executable_rate": round(ok / max(1, done), 4),
            "text_near_dup_rate": (
                round(near_dup_hits / max(1, near_dup_total), 4) if near_dup_total else None
            ),
        },
        "paths": {
            "results": str(results_path),
            "dir_models": str(models_base_dir),
            "dir_raw_outputs": str(raw_output_dir),
            "dir_raw_failures": str(raw_fail_dir),
            "dir_failures": str(fail_dir),
        },
        "notes": [
            "Novelty via MinHash/LSH against NN_Rag_gen_train.jsonl",
            "Near-duplicates rejected up to max_rejections",
        ],
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {ok}/{done} succeeded. Results → {results_path}")
    if near_dup_total > 0:
        print(
            f"[NOVELTY-TEXT] accepted near-dup hits: {near_dup_hits} / {near_dup_total}  "
            f"(τ={near_dup_threshold})"
        )
    print(f"Summary → {summary_path}")
    print(f"Generated models (NNEval-compatible) → {models_base_dir}")
    print(f"Raw outputs → {raw_output_dir}")
    print(f"Raw failures → {raw_fail_dir}")
    print(f"Failure code → {fail_dir}")
