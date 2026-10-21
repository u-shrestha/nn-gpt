"""The Tune command line must parse: every option is defined once (argparse raises on duplicates at start-up).

The module is read with ast, not imported, so the test needs neither torch nor the LEMUR database."""
import ast
from collections import Counter
from pathlib import Path

TUNE = Path(__file__).resolve().parents[1] / "ab" / "gpt" / "act" / "tune" / "Tune.py"


def _option_strings(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    yield arg.value


def test_every_cli_option_is_defined_once():
    counts = Counter(_option_strings(ast.parse(TUNE.read_text())))
    assert counts, "no add_argument calls found"
    duplicates = {opt: n for opt, n in counts.items() if n > 1}
    assert not duplicates, f"options defined more than once: {duplicates}"
