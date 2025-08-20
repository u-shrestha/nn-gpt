import ast
import io
import tokenize


def dedup_imports(src: str) -> str:
    """Merge duplicate torch import lines."""
    seen, out = set(), []
    for ln in src.splitlines():
        if ln.startswith("import torch") or ln.startswith("from torch import nn"):
            if ln in seen:
                continue
            seen.add(ln)
        out.append(ln)
    return "\n".join(out)


# ────────────────────────────────────────────────────────────────
#  Normalise stray top-level indentation (safe for half-finished files)
# ────────────────────────────────────────────────────────────────
def normalize_top_indent(src: str) -> str:
    try:
        ast.parse(src)
        return src
    except (SyntaxError, IndentationError):
        pass

    lines = src.splitlines(True)
    for i, ln in enumerate(lines):
        if ln.strip():
            lines[i] = ln.lstrip()
            break
    return "".join(lines)


def strip_comments(code: str) -> str:
    result = []
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)

    prev_toktype = tokenize.INDENT
    for tok in tokens:
        tok_type, tok_string, _, _, _ = tok

        if tok_type == tokenize.COMMENT:
            # skip comments
            continue
        elif tok_type == tokenize.STRING:
            # skip likely docstrings (standalone strings right after indent or at start)
            if prev_toktype == tokenize.INDENT or prev_toktype == tokenize.NEWLINE:
                continue

        result.append(tok)
        prev_toktype = tok_type

    return tokenize.untokenize(result).strip()


def improve_code(code: str) -> str:
    try:
        if code:
            return normalize_top_indent(dedup_imports(strip_comments(code)))
    except:
        pass
    return None
