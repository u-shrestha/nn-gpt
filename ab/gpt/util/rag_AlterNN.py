"""
•  Prepends each retrieved *block* verbatim to the LLM-generated wrapper so the
   saved file is self-contained.
•  Removes unused imports that reference packages **not installed** in the
   current Python environment (and never crashes on malformed source).
•  Normalises stray top-level indentation in both block and wrapper.
•  Retry feedback loop: gate traceback is fed back to the LLM; retries use
   sampling so the model can self-correct.
•  Hard 2 000 000 parameter budget to avoid OOMs at training time.
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import textwrap
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ab.rag.retriever import Retriever, BLOCKS_100
from ..util.Const import (
    conf_test_dir,
    epoch_dir,
    synth_dir,
    new_out_file,
    new_nn_file,
)
from ..util.Code import *
from ..util.Util import extract_code
from ab.nn.util.Util import create_file

# ────────────────────────────────────────────────────────────────
MAX_TRIES      = 3
LLM_MAX_TOKENS = 3_500
PARAM_BUDGET   = 2_000_000
random.seed(42)

blocks_dir = Path(__file__).resolve().parents[3] / "blocks"
blocks_dir.mkdir(exist_ok=True)

retriever = Retriever()

# ────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────
def _load_block(name: str) -> str | None:
    """Fetch a code block from cache or retriever."""
    p = blocks_dir / f"{name}.py"
    if p.is_file():
        print(f"[CACHE] {name}")
        return p.read_text()
    print(f"[FETCH] {name}")
    code = retriever.get_block(name)
    if code:
        p.write_text(code)
    return code


def _escape_braces(s: str) -> str:
    """Double every literal brace except {block} placeholder."""
    sent = "<<BLOCK>>"
    return s.replace("{block}", sent).replace("{", "{{").replace("}", "}}").replace(
        sent, "{block}"
    )


# ────────────────────────────────────────────────────────────────
#  1) strip unused imports of pkgs not installed
#  2) never crash on malformed blocks (TokenError, etc.)
# ────────────────────────────────────────────────────────────────
def _strip_unavailable_imports(src: str) -> tuple[str, bool]:
    """
    returns (clean, False) … nothing stripped / parse failed
            (clean, True)  … at least one dead import removed
            ("",    None)  … unavailable pkg is *used* → skip block
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, IndentationError, tokenize.TokenError):
        return src, False  # leave unchanged – later gates will decide

    dead, alias2pkg = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                top = al.name.split(".")[0]
                if importlib.util.find_spec(top) is None:
                    dead[node.lineno] = al.asname or top
                    alias2pkg[al.asname or top] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if importlib.util.find_spec(top) is None:
                for al in node.names:
                    dead[node.lineno] = al.asname or al.name
                    alias2pkg[al.asname or al.name] = top

    if not dead:
        return src, False

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        root = node
        if isinstance(node, ast.Attribute):
            while isinstance(root, ast.Attribute):
                root = root.value
        if isinstance(root, ast.Name) and root.id in alias2pkg:
            used.add(root.id)

    if used:
        return "", None  # real dependency on missing pkg

    cleaned = [
        ln for i, ln in enumerate(src.splitlines(True), 1) if i not in dead
    ]
    return "".join(cleaned), True


# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────
def alter(epochs: int, test_conf: str, llm_name: str) -> None:
    """Generate altered NNs using one building block per wrapper."""

    with open(conf_test_dir / test_conf) as f:
        template = json.load(f)

    prompt_tpl: List[str] | None = template.get("prompt")
    if prompt_tpl is None:
        for v in template.values():
            if isinstance(v, dict) and "prompt" in v:
                prompt_tpl = v["prompt"]
                break
    if prompt_tpl is None:
        raise KeyError("no 'prompt' key in template")

    print("Loading DeepSeek-Coder-7B-Instruct …")
    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    print("Ready ✔")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    kept = tried = 0

    for ep in range(epochs):
        queue = BLOCKS_100.copy()
        base_out = epoch_dir(ep)
        idx = 0

        while queue:
            name = queue.pop(random.randrange(len(queue)))
            raw_block = _load_block(name)
            if raw_block is None:
                continue

            block, flag = _strip_unavailable_imports(raw_block)
            if flag is None:
                print(f"[SKIP] {name}: needs unavailable package")
                continue
            if flag:
                print(f"[STRIP] {name}: removed unused imports of unavailable packages")

            block = normalize_top_indent(textwrap.dedent(block).lstrip("\n"))

            user_prompt = _escape_braces("\n".join(prompt_tpl)).format(block=block)
            if len(tok.tokenize(user_prompt)) + 50 > CONTEXT_WINDOW_LIMIT:
                print("[WARN] prompt near context window")

            prev_error = ""
            for attempt in range(MAX_TRIES):
                tried += 1
                sys_msg = f"{llm_name} – retry={attempt}"
                if attempt and prev_error:
                    sys_msg += f" | previous error: {prev_error}"

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_prompt},
                ]
                inp = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)

                gen = model.generate(
                    inp,
                    max_new_tokens=LLM_MAX_TOKENS,
                    temperature=(0.2 if attempt == 0 else 0.7),
                    top_p=0.9,
                    do_sample=(attempt > 0),
                    eos_token_id=tok.eos_token_id,
                )

                raw = tok.decode(gen[0][len(inp[0]):], skip_special_tokens=True)
                wrapper = textwrap.dedent(
                    (extract_code(raw) or "").replace("```python", "").replace("```", "")
                ).strip()

                full_code = dedup_imports(block + "\n\n" + wrapper)

                try:
                    compiled = compile(full_code, "<string>", "exec")
                    ns: dict[str, object] = {}
                    exec(compiled, ns)
                    Net = ns.get("Net")
                    shp = ns.get("supported_hyperparameters")
                    if Net is None or shp is None:
                        raise ValueError("missing Net or supported_hyperparameters")

                    dummy = {"lr": 0.01, "momentum": 0.9}
                    net = Net((3, 32, 32), (10,), dummy, torch.device("cpu"))
                    if sum(p.numel() for p in net.parameters()) > PARAM_BUDGET:
                        raise ValueError("param budget exceeded")

                except Exception as e:
                    prev_error = str(e)[:ERROR_MSG_TRUNC_LEN]
                    print(f"[GATE] {name} failed ({prev_error})")
                    continue  # retry

                break  # success
            else:
                print(f"[FAIL] giving up on {name}")
                continue

            out_dir = synth_dir(base_out) / f"B{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / new_nn_file).write_text(full_code)
            create_file(out_dir, new_out_file, raw)
            print(f"[INFO] ✔ {name} → {out_dir / new_nn_file}")

            idx += 1
            kept += 1

        print(f"\n[STATS] kept {kept} / {tried} generations ({kept/tried:.1%})")
