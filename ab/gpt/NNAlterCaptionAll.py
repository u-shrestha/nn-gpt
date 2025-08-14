# NNAlterCaptionAll.py
import argparse
import json
import tempfile
from pathlib import Path

from ab.gpt.util.AlterCaptionNN import alter
from ab.gpt.util.Const import conf_test_dir

CAPTION_PROMPT_FILE = "NN_caption_mix.json"

KEY_STRUCT = "improvement_captioning_structural"
KEY_NUM = "improvement_captioning_numeric"

def _filter_prompt_json(src_path: Path, keys_to_keep):
    """Create a temp JSON containing only selected keys. Returns temp filename (str)."""
    with open(src_path, "r") as f:
        data = json.load(f)
    filtered = {k: v for k, v in data.items() if k in keys_to_keep}
    if not filtered:
        raise ValueError(f"No matching keys in {src_path.name} for {keys_to_keep}")

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.write(json.dumps(filtered, ensure_ascii=False, indent=2).encode("utf-8"))
    tmp.flush()
    tmp.close()
    return tmp.name

def main():
    parser = argparse.ArgumentParser(description="Unified captioning alter runner (structural + numeric).")
    parser.add_argument("-e", "--epochs", type=int, default=8)
    parser.add_argument("-m", "--model", type=str,
                        default="deepseek-ai/deepseek-coder-7b-instruct-v1.5")
    parser.add_argument("--mode", choices=["structural", "numeric", "both"], default="both")
    parser.add_argument("-c", "--conf", type=str, default=CAPTION_PROMPT_FILE)
    parser.add_argument("--gguf", type=str, default=None)
    parser.add_argument("--final-dir", type=str,
        default="/home/krunaljesani/Downloads/ABrain-One/nn-dataset/ab/nn/nn")
    # NEW: target a single captioning model by name (matches the 'nn' column)
    parser.add_argument("--only-nn", type=str, default=None,
        help="Generate variants ONLY for this model name (e.g. RESNETLSTM, ResNetTransformer).")

    args = parser.parse_args()
    src_path = conf_test_dir / args.conf

    # choose keys based on mode
    if args.mode == "both":
        # Use the original JSON (contains both) — AlterNN.alter will iterate all keys
        conf_file_to_use = src_path.name
    else:
        keys = [KEY_STRUCT] if args.mode == "structural" else [KEY_NUM]
        temp_json = _filter_prompt_json(src_path, keys)
        # Pass a path relative to conf_test_dir if temp is elsewhere
        # AlterNN.alter expects conf_test_dir / filename, so give absolute by using a dummy name and symlink? Simpler: pass absolute via hacking?
        # We can temporarily copy: but easiest is to pass absolute path by making a temp in conf_test_dir.
        # For portability, just generate another temp in conf_test_dir.
        conf_file_to_use = Path(temp_json).name
        # Move/copy into conf_test_dir so AlterNN can find it
        dest = conf_test_dir / conf_file_to_use
        Path(temp_json).replace(dest)

    # Call existing pipeline
    alter(args.epochs, conf_file_to_use, args.model, gguf_file=args.gguf, final_out_dir=args.final_dir, only_nn=args.only_nn)

if __name__ == "__main__":
    main()
