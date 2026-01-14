import ab.gpt.TuneNNGen as TuneNNGen
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_run', action='store_true', default=False,
                        help='Enable ONNX mode')
    args, unknown = parser.parse_known_args()  # Parse only known args
    TuneNNGen.main(llm_conf='ds_coder_1.3b_instruct.json', max_new_tokens=4 * 1024,
    onnx_run=args.onnx_run
    # gradient_accumulation_steps=4
    )


if __name__ == '__main__':
    main()
