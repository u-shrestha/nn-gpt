import argparse

from ab.gpt.util.ModifyCode import modify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    args = parser.parse_args()
    modify(args.epochs, 'test_nn_chg_prompts_generation.json', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')


if __name__ == "__main__":
    main()
