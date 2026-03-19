import os
import ast
import hashlib
import json
import re
import shutil
import textwrap
from typing import Dict, List, Sequence, Tuple


RAW_HF_HOME = "/media/xi/Data/hf-cache"
RAW_HF_HUB_CACHE = "/media/xi/Data/hf-cache/hub"
RAW_TRANSFORMERS_CACHE = "/media/xi/Data/hf-cache/transformers"
RAW_LOG_DIR = "rl_output/raw"
RAW_EPOCH_ROOT = "out/nngpt/llm/epoch_raw"
RAW_TRAINER_OUT = "grpo_backbone_outputs/raw"
RAW_MAX_COMPLETION_LENGTH = "1536"
RAW_TEMPERATURE = "0.9"
RAW_NUM_GENERATIONS = "8"
RAW_GRAD_ACCUM = "8"

# Force the raw runtime to use the mounted Windows cache and stable raw output dirs.
os.environ["HF_HOME"] = RAW_HF_HOME
os.environ["HF_HUB_CACHE"] = RAW_HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = RAW_HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = RAW_TRANSFORMERS_CACHE
os.environ["NNGPT_RL_LOG_DIR"] = RAW_LOG_DIR
os.environ["NNGPT_RL_EPOCH_ROOT"] = RAW_EPOCH_ROOT
os.environ["NNGPT_RL_TRAINER_OUT"] = RAW_TRAINER_OUT
os.environ["NNGPT_RL_MAX_COMPLETION_LENGTH"] = RAW_MAX_COMPLETION_LENGTH
os.environ["NNGPT_RL_TEMPERATURE"] = RAW_TEMPERATURE
os.environ["NNGPT_RL_NUM_GENERATIONS"] = RAW_NUM_GENERATIONS
os.environ["NNGPT_RL_GRAD_ACCUM"] = RAW_GRAD_ACCUM

import ab.gpt.TuneRL as TuneRL
import ab.gpt.util.SFTUtil as SFTUtil


RAW_BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
RAW_MODEL_OUT = "rl_backbone_model_raw"

RAW_DISCOVERY_PROMPT_TEMPLATE = """
You are writing one novel image-classification architecture.

Return EXACTLY three XML blocks and nothing else.
- The first non-whitespace token must be `<block>`
- The last non-whitespace token must be `</forward>`
- Do not write markdown, explanations, or prose

Discovery Track
- Track Name: {goal_name}
- Discovery Target Tags: {target_tags}
- Design Brief: {design_brief}

Available runtime helpers already exist:
- `TorchVision(model=..., in_channels=...)`
- `FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)`
- `adaptive_pool_flatten(x)`
- `self.infer_dimensions_dynamically(out_shape[0])`

Backbones are optional. Prefer direct CNN / FractalBlock modules unless a backbone is essential.
If you explicitly use a backbone, choose 1-2 from:
[{available_backbones}]

Hard rules
1. `self.pattern` must be a NEW motif name, not one of: {legacy_patterns}
2. In `__init__`, set `self.device = device`, `self.use_amp = torch.cuda.is_available()`, and `self._input_spec = tuple(in_shape[1:])`
3. Call `self.infer_dimensions_dynamically(out_shape[0])`
4. `forward` must return classifier logits
5. Use `adaptive_pool_flatten(...)` before concatenation or classifier input
6. Do not use `if self.pattern` in `forward`
7. Avoid the plain one-shot Parallel_Triple topology
8. Use the target tags with actual graph structure, not only names
9. Do not define any new classes or helper methods besides the 3 required defs
10. Keep `forward` as straight-line assignments plus a final return
11. If unsure, still output the three XML blocks with best-effort valid Python
12. Prefer modules named like `stem`, `project_a`, `project_b`, `bridge`, `fractal`, `fuse`
13. Prefer plain CNN blocks or `FractalBlock`; avoid invented helper class names
14. Do not emit `import ...` lines or `class ...` definitions in the completion
15. Never define `DropConv3x3Block` or any wrapper class; write only the required defs

Helpful module names:
{module_hints}

The assistant response is already prefixed with:
`<block>`
`def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):`
Continue with the function body only, then close `</block>`, then emit `<init>...</init>` and `<forward>...</forward>`.

Implement exactly these signatures:
<block>
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    ...
</block>
<init>
def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
    ...
</init>
<forward>
def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    ...
</forward>
"""

ORIGINAL_EXTRACT_COMPLETION_BLOCKS = TuneRL.extract_completion_blocks
ORIGINAL_REWARD_FN = TuneRL.reward_fn

DEFAULT_BLOCK_CODE = """
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout_prob and dropout_prob > 0:
        layers.append(nn.Dropout2d(dropout_prob))
    return nn.Sequential(*layers)
""".strip()

BLOCKED_ATTRS = {
    "device",
    "use_amp",
    "_input_spec",
    "pattern",
    "classifier",
    "infer_dimensions_dynamically",
    "train_setup",
    "learn",
    "criterion",
    "optimizer",
    "_scaler",
}

PLACEHOLDER_PATTERN_NAMES = {
    "",
    "OpenMotif",
    "OpenMotifRaw",
    "NovelPattern",
    "NewPattern",
    "CustomPattern",
}

PROJECT_NAME_POOL = [
    "project_a",
    "project_b",
    "bridge",
    "project_c",
    "adapter",
]

FALLBACK_TEMPLATE_NAMES = (
    "stem_pair_fuse",
    "stem_bridge_wide_fuse",
    "stem_reuse_fuse",
    "project_fractal_merge",
    "project_reuse_mixer",
    "project_deep_cnn",
)

RAW_ASSISTANT_PREFIX = (
    "<block>\n"
    "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):\n"
)

EXTRACTION_META_CACHE: Dict[str, Dict[str, object]] = {}


class RawCodeLogger(TuneRL.SimpleCodeLogger):
    def __init__(self, output_dir: str = "rl_output/raw"):
        super().__init__(output_dir)
        self.samples_file = os.path.join(output_dir, "generation_samples.jsonl")

    def log_generation(self, prompt: str, completion: str, reward: float, api_result=None):
        super().log_generation(prompt, completion, reward, api_result)
        record = {
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "api_result": api_result,
        }
        with open(self.samples_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def configure_raw_defaults() -> None:
    # Keep raw experiments isolated from the SFT trainer outputs with code-owned defaults.
    os.environ["HF_HOME"] = RAW_HF_HOME
    os.environ["HF_HUB_CACHE"] = RAW_HF_HUB_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = RAW_HF_HUB_CACHE
    os.environ["TRANSFORMERS_CACHE"] = RAW_TRANSFORMERS_CACHE
    os.environ["NNGPT_RL_LOG_DIR"] = RAW_LOG_DIR
    os.environ["NNGPT_RL_EPOCH_ROOT"] = RAW_EPOCH_ROOT
    os.environ["NNGPT_RL_TRAINER_OUT"] = RAW_TRAINER_OUT
    os.environ["NNGPT_RL_MODEL_OUT"] = RAW_MODEL_OUT
    os.environ["NNGPT_RL_MAX_COMPLETION_LENGTH"] = RAW_MAX_COMPLETION_LENGTH
    os.environ["NNGPT_RL_TEMPERATURE"] = RAW_TEMPERATURE
    os.environ["NNGPT_RL_NUM_GENERATIONS"] = RAW_NUM_GENERATIONS
    os.environ["NNGPT_RL_GRAD_ACCUM"] = RAW_GRAD_ACCUM


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return TuneRL.clean_block(match.group(1)) if match else ""


def _strip_outer_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_function_block(text: str, fn_name: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.match(rf"^\s*def {re.escape(fn_name)}\s*\(", line):
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for j in range(i + 1, len(lines)):
            stripped = lines[j].lstrip()
            if not stripped:
                continue
            indent_j = len(lines[j]) - len(stripped)
            if (stripped.startswith("def ") or stripped.startswith("class ")) and indent_j <= indent:
                end = j
                break
        return textwrap.dedent("\n".join(lines[i:end])).strip()
    return ""


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _clean_source_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("<s=", "=")
    text = text.replace("torch.concat(", "torch.cat(")
    text = text.replace("torch.concatenate(", "torch.cat(")
    text = text.replace("self.adaptive_pool_flatten(", "adaptive_pool_flatten(")
    return text.strip()


def _completion_cache_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _hash_pick(seed_text: str, options: Sequence[str]) -> str:
    if not options:
        return ""
    digest = hashlib.sha1((seed_text or "raw").encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _extract_pattern_name(*texts: str) -> str:
    for text in texts:
        if not text:
            continue
        match = re.search(r"self\.pattern\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            pattern_name = match.group(1).strip()
            if pattern_name not in PLACEHOLDER_PATTERN_NAMES:
                return pattern_name
    return ""


def _infer_attr_role(attr_name: str) -> str:
    lowered = attr_name.lower()
    if "fractal" in lowered:
        return "fractal"
    if lowered.startswith("backbone"):
        return "backbone"
    if "stem" in lowered:
        return "stem"
    if any(token in lowered for token in ("project", "bridge", "adapter", "align")):
        return "project"
    if any(token in lowered for token in ("fuse", "merge", "gate", "mixer")):
        return "fuse"
    return "generic"


def _has_structural_attr(attrs: Sequence[str]) -> bool:
    return any(
        _infer_attr_role(attr) in {"stem", "project", "fuse", "backbone", "fractal"}
        for attr in attrs
    )


def _module_ctor(attr_name: str) -> str:
    return _module_ctor_with_input(attr_name, "48")


def _module_ctor_with_input(attr_name: str, in_channels_expr: str) -> str:
    role = _infer_attr_role(attr_name)
    if role == "stem":
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "drop_conv3x3_block(48, 48, dropout_prob=dropout_prob)"
            ")"
        )
    if role == "fractal":
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "FractalBlock(48, 48, num_columns=3, loc_drop_prob=0.0, dropout_prob=dropout_prob)"
            ")"
        )
    if role == "fuse" or "bridge" in attr_name.lower():
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "drop_conv3x3_block(48, 48, dropout_prob=dropout_prob)"
            ")"
        )
    return f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob)"


def _collect_completion_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    attrs = _dedupe_keep_order(attrs)
    if not attrs:
        return ["stem", "project_a", "project_b", "fuse"]
    if not _has_structural_attr(attrs):
        return ["stem", *attrs[:2], "fuse"]
    return attrs


def _scan_raw_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    return _dedupe_keep_order(attrs)


def _normalize_attr_list(attrs: Sequence[str]) -> List[str]:
    normalized = [
        attr
        for attr in _dedupe_keep_order(list(attrs))
        if attr and attr not in BLOCKED_ATTRS and not attr.startswith("__")
    ]
    if not normalized:
        return ["stem", "project_a", "project_b", "fuse"]
    if not _has_structural_attr(normalized):
        return _dedupe_keep_order(["stem", *normalized[:2], "fuse"])
    return normalized


def _candidate_body_lines(text: str, fn_name: str) -> List[str]:
    cleaned = _clean_source_text(text)
    fn_block = _extract_function_block(cleaned, fn_name)
    source = fn_block or cleaned
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith(f"def {fn_name}"):
        lines = lines[1:]

    body: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith(("<", ">", "```", "class ", "def ", "from ", "import ")):
            continue
        if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except", "finally:")):
            continue
        if "self.pattern" in stripped:
            continue
        if stripped in {"pass", "..."}:
            continue
        body.append(stripped)
    return body


def _register_attr(attr_name: str, used_attrs: List[str]) -> None:
    if attr_name in BLOCKED_ATTRS or attr_name.startswith("__"):
        return
    if attr_name not in used_attrs:
        used_attrs.append(attr_name)


def _make_unique_attr_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    suffix = 2
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    unique_name = f"{base_name}_{suffix}"
    used_names.add(unique_name)
    return unique_name


def _canonical_attr_mapping(attrs: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used_names: set[str] = set()
    project_idx = 0
    backbone_idx = 0

    for attr in _normalize_attr_list(attrs):
        lowered = attr.lower()
        role = _infer_attr_role(attr)
        if role == "stem":
            base_name = "stem"
        elif role == "fuse":
            base_name = "fuse"
        elif role == "fractal":
            base_name = "fractal"
        elif role == "project":
            if "bridge" in lowered:
                base_name = "bridge"
            elif project_idx < len(PROJECT_NAME_POOL):
                base_name = PROJECT_NAME_POOL[project_idx]
                project_idx += 1
            else:
                base_name = f"project_{project_idx - len(PROJECT_NAME_POOL) + 4}"
                project_idx += 1
        elif role == "backbone":
            base_name = f"backbone_{chr(ord('a') + backbone_idx)}"
            backbone_idx += 1
        else:
            if project_idx < len(PROJECT_NAME_POOL):
                base_name = PROJECT_NAME_POOL[project_idx]
                project_idx += 1
            else:
                base_name = f"project_{project_idx - len(PROJECT_NAME_POOL) + 4}"
                project_idx += 1

        mapping[attr] = _make_unique_attr_name(base_name, used_names)
    return mapping


def _rename_expr_tokens(expr: str, mapping: Dict[str, str]) -> str:
    renamed = expr
    for old_name in sorted(mapping, key=len, reverse=True):
        renamed = re.sub(rf"\b{re.escape(old_name)}\b", mapping[old_name], renamed)
    return renamed


def _canonicalize_assignments(
    assignments: Sequence[Tuple[str, str]],
    used_attrs: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    attr_mapping = _canonical_attr_mapping(used_attrs)
    renamed_assignments = [
        (lhs, _rename_expr_tokens(expr, {f"self.{k}": f"self.{v}" for k, v in attr_mapping.items()}))
        for lhs, expr in assignments
    ]

    compacted: List[Tuple[str, str]] = []
    var_mapping: Dict[str, str] = {"x": "x"}
    known_vars = {"x"}
    for lhs, expr in renamed_assignments:
        expr = _rename_expr_tokens(expr, var_mapping)
        if expr in known_vars:
            var_mapping[lhs] = expr
            continue
        new_lhs = f"x{len(compacted)}"
        compacted.append((new_lhs, expr))
        var_mapping[lhs] = new_lhs
        known_vars.add(new_lhs)

    canonical_attrs = _dedupe_keep_order([attr_mapping[attr] for attr in used_attrs if attr in attr_mapping])
    return compacted, canonical_attrs


def _channel_sum(parts: Sequence[str]) -> str:
    numeric_total = 0
    symbolic_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\d+", part):
            numeric_total += int(part)
        else:
            symbolic_parts.append(part)
    if numeric_total:
        symbolic_parts.append(str(numeric_total))
    if not symbolic_parts:
        return "48"
    if len(symbolic_parts) == 1:
        return symbolic_parts[0]
    return " + ".join(symbolic_parts)


def _infer_expr_channels(
    node: ast.AST,
    var_channels: Dict[str, str],
    attr_inputs: Dict[str, str],
) -> str:
    if isinstance(node, ast.Name):
        return var_channels.get(node.id, "48")

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _infer_expr_channels(node.left, var_channels, attr_inputs)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "self":
                attr_name = node.func.attr
                child_channels = (
                    _infer_expr_channels(node.args[0], var_channels, attr_inputs)
                    if node.args else "48"
                )
                attr_inputs.setdefault(attr_name, child_channels)
                return "48"

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = f"{node.func.value.id}.{node.func.attr}"

        if func_name in {"torch.cat", "cat"} and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            child_channels = [
                _infer_expr_channels(item, var_channels, attr_inputs)
                for item in node.args[0].elts
            ]
            return _channel_sum(child_channels)

        if node.args:
            return _infer_expr_channels(node.args[0], var_channels, attr_inputs)

    return "48"


def _infer_attr_input_channels(assignments: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    var_channels: Dict[str, str] = {"x": "in_shape[1]"}
    attr_inputs: Dict[str, str] = {}
    for lhs, expr in assignments:
        try:
            node = ast.parse(expr, mode="eval").body
            var_channels[lhs] = _infer_expr_channels(node, var_channels, attr_inputs)
        except Exception:
            var_channels[lhs] = "48"
    return attr_inputs


def _derive_pattern_name(
    attrs: Sequence[str],
    assignments: Sequence[Tuple[str, str]],
    completion_text: str,
) -> str:
    role_tokens: List[str] = []
    if any(_infer_attr_role(attr) == "stem" for attr in attrs):
        role_tokens.append("Stem")
    if any(_infer_attr_role(attr) == "project" for attr in attrs):
        role_tokens.append("Project")
    if any(_infer_attr_role(attr) == "fractal" for attr in attrs):
        role_tokens.append("Fractal")
    if any("torch.cat(" in expr or " + " in expr for _, expr in assignments):
        role_tokens.append("Merge")
    if any(_infer_attr_role(attr) == "fuse" for attr in attrs):
        role_tokens.append("Fuse")
    base_name = "".join(role_tokens[:4]) or "OpenMotif"
    digest_source = "||".join([*attrs, *[expr for _, expr in assignments], completion_text])
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:6].upper()
    return f"{base_name}_{digest}"


def _render_expr(
    node: ast.AST,
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    if isinstance(node, ast.Name):
        if node.id in var_aliases:
            return var_aliases[node.id]
        return node.id if node.id in known_vars or node.id == "x" else "x"

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return repr(node.value)
        return ""

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        _register_attr(node.attr, used_attrs)
        return f"self.{node.attr}"

    if isinstance(node, ast.IfExp):
        body_expr = _render_expr(node.body, known_vars, used_attrs, var_aliases)
        else_expr = _render_expr(node.orelse, known_vars, used_attrs, var_aliases)
        if body_expr and "self." in body_expr:
            return body_expr
        if else_expr:
            return else_expr
        return body_expr

    if isinstance(node, ast.List):
        items = [_render_expr(item, known_vars, used_attrs, var_aliases) for item in node.elts]
        items = [item for item in items if item]
        return ", ".join(items)

    if isinstance(node, ast.Tuple):
        items = [_render_expr(item, known_vars, used_attrs, var_aliases) for item in node.elts]
        items = [item for item in items if item]
        return ", ".join(items)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _render_expr(node.left, known_vars, used_attrs, var_aliases)
        right = _render_expr(node.right, known_vars, used_attrs, var_aliases)
        if left and right:
            return f"({left} + {right})"
        return left or right

    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name):
                func_name = f"{base.id}.{node.func.attr}"
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                func_name = f"self.{base.attr}.{node.func.attr}"

        if func_name in {"adaptive_pool_flatten", "pool_flatten"}:
            child = _render_expr(node.args[0], known_vars, used_attrs, var_aliases) if node.args else "x"
            return child

        if func_name in {"torch.cat", "cat", "torch.stack", "stack"}:
            parts: List[str] = []
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                for item in node.args[0].elts:
                    rendered = _render_expr(item, known_vars, used_attrs, var_aliases)
                    if rendered:
                        parts.append(rendered)
            else:
                for item in node.args:
                    rendered = _render_expr(item, known_vars, used_attrs, var_aliases)
                    if rendered:
                        parts.append(rendered)
            parts = _dedupe_keep_order(parts)
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return f"torch.cat([{', '.join(parts)}], dim=1)"

        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "self":
                attr_name = node.func.attr
                _register_attr(attr_name, used_attrs)
                child = _merge_call_args(node.args, known_vars, used_attrs, var_aliases)
                return f"self.{attr_name}({child})"

            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                attr_name = base.attr
                _register_attr(attr_name, used_attrs)
                child = _render_expr(node.args[0], known_vars, used_attrs, var_aliases) if node.args else "x"
                if node.func.attr in {"flatten", "view", "reshape", "contiguous", "mean"}:
                    return f"self.{attr_name}({child})"
                return f"self.{attr_name}({child})"

            child = _render_expr(base, known_vars, used_attrs, var_aliases)
            if node.func.attr in {"flatten", "view", "reshape", "contiguous", "mean"}:
                return child
            if node.func.attr in {"float", "to", "cpu", "cuda"}:
                return child

        if node.args:
            return _render_expr(node.args[0], known_vars, used_attrs, var_aliases)
        return ""

    return ""


def _merge_call_args(
    args: Sequence[ast.AST],
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    rendered_args = [
        _render_expr(arg, known_vars, used_attrs, var_aliases)
        for arg in args
    ]
    rendered_args = [arg for arg in rendered_args if arg]
    rendered_args = _dedupe_keep_order(rendered_args)
    if not rendered_args:
        return "x"
    if len(rendered_args) == 1:
        return rendered_args[0]
    return f"torch.cat([{', '.join(rendered_args)}], dim=1)"


def _sanitize_expr_text(
    expr_text: str,
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    expr_text = expr_text.strip().rstrip(",")
    if not expr_text:
        return ""
    expr_text = expr_text.replace("self.adaptive_pool_flatten(", "adaptive_pool_flatten(")
    expr_text = expr_text.replace("torch.concat(", "torch.cat(")
    expr_text = expr_text.replace("torch.concatenate(", "torch.cat(")
    try:
        node = ast.parse(expr_text, mode="eval").body
        rendered = _render_expr(node, known_vars, used_attrs, var_aliases)
        if rendered:
            return rendered
    except Exception:
        pass

    calls = re.findall(r"self\.([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)", expr_text)
    parts = []
    for attr_name, var_name in calls:
        if attr_name in BLOCKED_ATTRS:
            continue
        _register_attr(attr_name, used_attrs)
        arg_name = var_aliases.get(var_name, var_name if var_name in known_vars or var_name == "x" else "x")
        parts.append(f"self.{attr_name}({arg_name})")

    if "torch.cat" in expr_text and parts:
        merged = _dedupe_keep_order(parts)
        if len(merged) == 1:
            return merged[0]
        return f"torch.cat([{', '.join(merged)}], dim=1)"

    if "+" in expr_text and len(parts) >= 2:
        return " + ".join(_dedupe_keep_order(parts[:3]))

    if parts:
        return parts[0]

    for name in reversed(list(known_vars)):
        if name in expr_text:
            return name
    return "x"


def _fallback_blueprint(template_name: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    if template_name == "stem_bridge_wide_fuse":
        attrs = ["stem", "project_a", "project_b", "bridge", "fuse"]
        return attrs, [
            ("x0", "self.stem(x)"),
            ("x1", "self.project_a(x0)"),
            ("x2", "self.project_b(x0)"),
            ("x3", "self.bridge(x1)"),
            ("x4", "torch.cat([x1, x2, x3], dim=1)"),
            ("x5", "self.fuse(x4)"),
        ]
    if template_name == "stem_reuse_fuse":
        attrs = ["stem", "project_a", "bridge", "fuse"]
        return attrs, [
            ("x0", "self.stem(x)"),
            ("x1", "self.project_a(x0)"),
            ("x2", "self.bridge(x1)"),
            ("x3", "self.fuse(torch.cat([x0, x2], dim=1))"),
        ]
    if template_name == "project_fractal_merge":
        attrs = ["project_a", "bridge", "fractal", "fractal_aux", "fuse"]
        return attrs, [
            ("x0", "self.project_a(x)"),
            ("x1", "self.bridge(x0)"),
            ("x2", "self.fractal(x1)"),
            ("x3", "self.fractal_aux(x2)"),
            ("x4", "torch.cat([x0, x1, x2, x3], dim=1)"),
            ("x5", "self.fuse(x4)"),
        ]
    if template_name == "project_reuse_mixer":
        attrs = ["project_a", "bridge", "adapter", "fuse"]
        return attrs, [
            ("x0", "self.project_a(x)"),
            ("x1", "self.bridge(x0)"),
            ("x2", "self.adapter(x1)"),
            ("x3", "torch.cat([x0, x2], dim=1)"),
            ("x4", "self.fuse(x3)"),
        ]
    if template_name == "project_deep_cnn":
        attrs = ["project_a", "bridge", "project_b", "project_c", "fuse"]
        return attrs, [
            ("x0", "self.project_a(x)"),
            ("x1", "self.bridge(x0)"),
            ("x2", "self.project_b(x1)"),
            ("x3", "self.project_c(x2)"),
            ("x4", "self.fuse(x3)"),
        ]
    attrs = ["stem", "project_a", "project_b", "fuse"]
    return attrs, [
        ("x0", "self.stem(x)"),
        ("x1", "self.project_a(x0)"),
        ("x2", "self.project_b(x0)"),
        ("x3", "torch.cat([x1, x2], dim=1)"),
        ("x4", "self.fuse(x3)"),
    ]


def _fallback_assignments(
    attrs: Sequence[str],
    seed_text: str,
) -> Tuple[List[Tuple[str, str]], List[str], str]:
    attrs = _normalize_attr_list(attrs)
    roles = {_infer_attr_role(attr) for attr in attrs}
    lowered = " ".join(attrs).lower()

    if not _has_structural_attr(attrs):
        options = list(FALLBACK_TEMPLATE_NAMES)
    elif "fractal" in roles or "fractal" in lowered:
        options = ["project_fractal_merge", "stem_reuse_fuse", "stem_bridge_wide_fuse"]
    elif "stem" in roles:
        options = ["stem_pair_fuse", "stem_bridge_wide_fuse", "stem_reuse_fuse"]
    elif any(token in lowered for token in ("bridge", "adapter", "mixer")):
        options = ["project_reuse_mixer", "project_deep_cnn", "project_fractal_merge"]
    else:
        options = ["project_deep_cnn", "project_reuse_mixer", "stem_pair_fuse"]

    template_name = _hash_pick(f"{seed_text}\n{'|'.join(attrs)}", options)
    template_attrs, assignments = _fallback_blueprint(template_name)
    return assignments, template_attrs, template_name


def _synthesize_forward_code(
    forward_code: str,
    *extra_texts: str,
) -> Tuple[str, List[str], List[Tuple[str, str]], Dict[str, object]]:
    candidate_lines = _candidate_body_lines(forward_code, "forward")
    used_attrs: List[str] = []
    known_vars = ["x"]
    var_aliases: dict[str, str] = {"x": "x"}
    assignments: List[Tuple[str, str]] = []
    synth_meta: Dict[str, object] = {
        "used_fallback": False,
        "fallback_template": "",
        "candidate_line_count": len(candidate_lines),
        "parsed_assignment_count": 0,
        "parsed_structural_attr_detected": False,
    }

    for raw_line in candidate_lines:
        if "self.classifier" in raw_line:
            continue

        if raw_line.startswith("return "):
            rendered = _sanitize_expr_text(raw_line[len("return "):], known_vars, used_attrs, var_aliases)
            continue

        match = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", raw_line)
        if not match:
            continue
        source_lhs, rhs = match.groups()
        rendered = _sanitize_expr_text(rhs, known_vars, used_attrs, var_aliases)
        if not rendered:
            continue
        target_name = f"x{len(assignments)}"
        assignments.append((target_name, rendered))
        known_vars.append(target_name)
        var_aliases[source_lhs] = target_name

    synth_meta["parsed_assignment_count"] = len(assignments)
    synth_meta["parsed_structural_attr_detected"] = _has_structural_attr(used_attrs)
    seed_text = "\n".join([forward_code, *extra_texts])
    if not assignments or not _has_structural_attr(used_attrs):
        attr_candidates = _collect_completion_attrs(forward_code, *extra_texts)
        assignments, used_attrs, fallback_template = _fallback_assignments(attr_candidates, seed_text)
        synth_meta["used_fallback"] = True
        synth_meta["fallback_template"] = fallback_template
    assignments, used_attrs = _canonicalize_assignments(assignments, used_attrs)
    if not assignments:
        fallback_attrs = ["stem", "project_a", "project_b", "fuse"]
        assignments, used_attrs, fallback_template = _fallback_assignments(fallback_attrs, seed_text or "raw")
        assignments, used_attrs = _canonicalize_assignments(assignments, used_attrs)
        synth_meta["used_fallback"] = True
        synth_meta["fallback_template"] = fallback_template
    last_expr = assignments[-1][0]

    forward_lines = [
        "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:",
    ]
    for lhs, expr in assignments:
        forward_lines.append(f"    {lhs} = {expr}")
    forward_lines.append(f"    feat = adaptive_pool_flatten({last_expr})")
    forward_lines.append("    if is_probing:")
    forward_lines.append("        return feat")
    forward_lines.append("    return self.classifier(feat)")
    synth_meta["final_assignment_count"] = len(assignments)
    synth_meta["used_attrs"] = list(_dedupe_keep_order(used_attrs))
    return "\n".join(forward_lines).strip(), _dedupe_keep_order(used_attrs), assignments, synth_meta


def _synthesize_init_code(
    attr_order: Sequence[str],
    pattern_name: str,
    assignments: Sequence[Tuple[str, str]],
) -> str:
    attrs = _normalize_attr_list(attr_order)
    attr_inputs = _infer_attr_input_channels(assignments)
    lines = [
        "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:",
        "    super().__init__()",
        f"    self.pattern = {pattern_name!r}",
        "    self.device = device",
        "    self.use_amp = torch.cuda.is_available()",
        "    self._input_spec = tuple(in_shape[1:])",
        "    dropout_prob = float(prm.get('dropout', prm.get('dropout_prob', 0.0)))",
    ]
    for attr_name in attrs:
        if attr_name in BLOCKED_ATTRS:
            continue
        input_channels = attr_inputs.get(attr_name, "in_shape[1]" if attr_name == "stem" else "48")
        lines.append(f"    self.{attr_name} = {_module_ctor_with_input(attr_name, input_channels)}")
    lines.append("    self.infer_dimensions_dynamically(out_shape[0])")
    return "\n".join(lines).strip()


def _normalize_block_code(_block_code: str) -> str:
    return DEFAULT_BLOCK_CODE


def _build_extraction_meta(
    completion: str,
    candidate: str,
    init_code: str,
    forward_code: str,
    used_attrs: Sequence[str],
    assignments: Sequence[Tuple[str, str]],
    synth_meta: Dict[str, object],
) -> Dict[str, object]:
    text = completion or ""
    xml_tag_count = sum(bool(_extract_xml_tag(text, tag)) for tag in ("block", "init", "forward"))
    class_count = len(re.findall(r"^\s*class\s+\w+", candidate, re.MULTILINE))
    import_count = len(re.findall(r"^\s*(?:from|import)\s+\w+", candidate, re.MULTILINE))
    bad_signature_count = len(re.findall(r"\)\s*-\s*:", candidate))
    raw_attrs = _scan_raw_attrs(text, candidate, init_code, forward_code)
    structural_attr_detected = _has_structural_attr(raw_attrs)
    quality_score = 0
    quality_score += xml_tag_count
    quality_score += 2 if structural_attr_detected else 0
    quality_score += 1 if synth_meta.get("parsed_assignment_count", 0) >= 3 else 0
    quality_score -= min(class_count, 2)
    quality_score -= min(import_count, 2)
    quality_score -= min(bad_signature_count, 2)
    if synth_meta.get("used_fallback") and not structural_attr_detected:
        quality_score -= 1
    low_evidence_fallback = bool(
        synth_meta.get("used_fallback") and not structural_attr_detected and quality_score <= 1
    )
    return {
        "xml_tag_count": xml_tag_count,
        "class_count": class_count,
        "import_count": import_count,
        "bad_signature_count": bad_signature_count,
        "structural_attr_detected": structural_attr_detected,
        "quality_score": quality_score,
        "used_fallback": bool(synth_meta.get("used_fallback")),
        "fallback_template": str(synth_meta.get("fallback_template") or ""),
        "parsed_assignment_count": int(synth_meta.get("parsed_assignment_count") or 0),
        "candidate_line_count": int(synth_meta.get("candidate_line_count") or 0),
        "used_attrs": list(used_attrs),
        "assignment_count": len(assignments),
        "low_evidence_fallback": low_evidence_fallback,
    }


def extract_completion_payload_tolerant(completion: str) -> Tuple[Tuple[str, str, str], Dict[str, object]]:
    block_code, init_code, forward_code = ORIGINAL_EXTRACT_COMPLETION_BLOCKS(completion)
    text = completion or ""
    cache_key = _completion_cache_key(text)
    cached = EXTRACTION_META_CACHE.get(cache_key)
    if cached:
        return (
            (cached["block_code"], cached["init_code"], cached["forward_code"]),
            dict(cached["meta"]),
        )

    block_code = block_code or _extract_xml_tag(text, "block")
    init_code = init_code or _extract_xml_tag(text, "init")
    forward_code = forward_code or _extract_xml_tag(text, "forward")

    candidate = _strip_outer_code_fences(text)
    if not (init_code and forward_code):
        parsed_block, parsed_init, parsed_forward = SFTUtil.parse_nn_code(candidate)
        block_code = block_code or TuneRL.clean_block(parsed_block or "")
        init_code = init_code or TuneRL.clean_block(parsed_init or "")
        forward_code = forward_code or TuneRL.clean_block(parsed_forward or "")

    if not init_code:
        init_code = TuneRL.clean_block(_extract_function_block(candidate, "__init__"))
    if not forward_code:
        forward_code = TuneRL.clean_block(_extract_function_block(candidate, "forward"))
    if not block_code:
        block_code = TuneRL.clean_block(_extract_function_block(candidate, "drop_conv3x3_block"))

    init_source = init_code or candidate
    forward_source = forward_code or candidate
    extracted_pattern_name = _extract_pattern_name(init_code, forward_code, candidate)
    synthesized_forward, used_attrs, assignments, synth_meta = _synthesize_forward_code(
        forward_source,
        init_source,
        candidate,
    )
    pattern_name = extracted_pattern_name or _derive_pattern_name(used_attrs, assignments, candidate)
    synthesized_init = _synthesize_init_code(used_attrs, pattern_name, assignments)
    normalized_block = _normalize_block_code(block_code)
    meta = _build_extraction_meta(
        text,
        candidate,
        init_source,
        forward_source,
        used_attrs,
        assignments,
        synth_meta,
    )
    EXTRACTION_META_CACHE[cache_key] = {
        "block_code": normalized_block,
        "init_code": synthesized_init,
        "forward_code": synthesized_forward,
        "meta": meta,
    }
    return ((normalized_block, synthesized_init, synthesized_forward), meta)


def extract_completion_blocks_tolerant(completion: str) -> Tuple[str, str, str]:
    blocks, _ = extract_completion_payload_tolerant(completion)
    return blocks


def extract_completion_meta(completion: str) -> Dict[str, object]:
    _, meta = extract_completion_payload_tolerant(completion)
    return meta


def raw_reward_fn(
    completion: str,
    *,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
):
    res = ORIGINAL_REWARD_FN(
        completion,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
    )
    meta = extract_completion_meta(completion)
    raw_delta = 0.0

    raw_delta -= 0.35 * min(int(meta.get("class_count", 0)), 2)
    raw_delta -= 0.12 * min(int(meta.get("import_count", 0)), 2)
    raw_delta -= 0.35 * min(int(meta.get("bad_signature_count", 0)), 2)

    xml_tag_count = int(meta.get("xml_tag_count", 0))
    if xml_tag_count < 3:
        raw_delta -= 0.18 * (3 - xml_tag_count)

    if meta.get("used_fallback"):
        raw_delta -= 0.20 if meta.get("structural_attr_detected") else 0.95

    if graph_info and graph_info.parse_ok:
        same_graph_count = batch_graph_hashes.count(graph_info.graph_hash) if batch_graph_hashes else 0
        same_family_count = batch_family_hashes.count(graph_info.family_hash) if batch_family_hashes else 0
        archive_family_freq = TuneRL.family_hash_archive_counts.get(graph_info.family_hash, 0)

        if same_graph_count > 1:
            raw_delta -= 0.75 * (same_graph_count - 1)
        if same_family_count > 1:
            raw_delta -= 0.95 * (same_family_count - 1)
        if same_family_count >= 4:
            raw_delta -= 1.20
        if same_family_count >= 8:
            raw_delta -= 2.00
        if archive_family_freq >= 1:
            raw_delta -= 0.30 * archive_family_freq

        if graph_info.fractal_calls >= 2:
            raw_delta += 0.35
        elif graph_info.fractal_calls == 1 and "fractal_deep" in (prompt_goal_tags or []):
            raw_delta += 0.10
        if graph_info.max_fan_in >= 3:
            raw_delta += 0.22
        if graph_info.merges >= 2 and graph_info.project_calls >= 1:
            raw_delta += 0.18
        if "fractal_deep" in (prompt_goal_tags or []) and graph_info.fractal_calls == 0:
            raw_delta -= 0.55
        if "wide_fuse" in (prompt_goal_tags or []) and graph_info.max_fan_in < 3:
            raw_delta -= 0.40
        if "branch_reuse" in (prompt_goal_tags or []) and graph_info.merges < 2:
            raw_delta -= 0.35

    res["reward"] = float(res.get("reward", -2.0)) + raw_delta
    if meta.get("low_evidence_fallback"):
        res["reward"] = min(float(res["reward"]), 0.0)

    res["raw_extraction"] = {
        **meta,
        "raw_delta": raw_delta,
    }
    return res


def load_rl_dataset_raw(tokenizer):
    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for RL")

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = row.get("accuracy", 0.8)
        for profile in goal_profiles:
            user_prompt = RAW_DISCOVERY_PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                design_brief=profile["brief"],
                module_hints=", ".join(profile["module_hints"]),
            )

            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": RAW_ASSISTANT_PREFIX},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            prompts.append(
                {
                    "prompt": prompt_str,
                    "accuracy": accuracy,
                    "goal_name": profile["name"],
                    "target_tags": ", ".join(profile["tags"]),
                }
            )

    return TuneRL.Dataset.from_list(prompts).shuffle(seed=42)


def patch_raw_runtime() -> None:
    TuneRL.base_model = RAW_BASE_MODEL
    TuneRL.SAVED_MODEL_PATH = RAW_MODEL_OUT
    TuneRL.PROMPT_TEMPLATE = RAW_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = extract_completion_blocks_tolerant
    TuneRL.reward_fn = raw_reward_fn
    TuneRL.load_rl_dataset = load_rl_dataset_raw


def bootstrap_raw_runtime() -> None:
    EXTRACTION_META_CACHE.clear()
    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    TuneRL.code_logger = RawCodeLogger(log_dir)

    print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
    shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)


def main() -> None:
    configure_raw_defaults()
    patch_raw_runtime()
    bootstrap_raw_runtime()
    TuneRL.main()


if __name__ == "__main__":
    main()
