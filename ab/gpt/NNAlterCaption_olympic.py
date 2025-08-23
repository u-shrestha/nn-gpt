
# ab/gpt/NNAlterCaptionAll.py
import argparse
from pathlib import Path

from ab.gpt.util.AlterCaptionNN import alter
from ab.gpt.util.Const import conf_test_dir


def main():
    parser = argparse.ArgumentParser(description="Generate revised captioning models (simple mode).")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of generation epochs to run.")
    parser.add_argument("-c", "--conf", type=str, default="NN_caption_mix.json")
    parser.add_argument("-nn", type=str, default=None, help="Generate variants ONLY for this model (e.g. RESNETLSTM, ResNetTransformer) otherwise default all.")
    parser.add_argument("-m", "--model", type=str, default="deepseek-ai/deepseek-coder-7b-instruct-v1.5", help="HF model id or local model path.")
    parser.add_argument("--gguf", type=str, default=None, help="Optional GGUF file path if using a local/llama.cpp backend.")

    args = parser.parse_args()

    # Resolve the config file (no temp files, no filtering)
    conf_path = conf_test_dir / args.conf
    if not conf_path.exists():
        raise FileNotFoundError(f"Config not found: {conf_path}")

    # Call the caption-alter pipeline directly
    alter(epochs=args.epochs,
        test_conf=conf_path.name,   # AlterCaptionNN expects a filename in conf_test_dir
        llm_name=args.model,
        gguf_file=args.gguf,
        only_nn=args.nn
    )


if __name__ == "__main__":
    main()
