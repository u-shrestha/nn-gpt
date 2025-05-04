import argparse
from ab.gpt.util.Tune import tune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip', type=int, default=-1, help='Number of epoches to skip the neural network generation.')
    parser.add_argument('-p', '--peft', type=str, default=None, help='Path to saved LoRA layers.')
    args = parser.parse_args()
    tune(10, 1, args.skip, args.peft, 'NN_gen.json', 'NN_gen.json', 'improve_classification_only',
         'nngpt_ds_coder_1.3b_instruct.json',
         n_training_prompts=None)

if __name__ == '__main__':
    main()
