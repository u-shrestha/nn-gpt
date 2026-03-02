"""
TuneNNGen Pipeline Wrapper
==========================
Simple wrapper to trigger the iterative fine-tuning pipeline
via TuneNNGen_v2.main() with run_iterative_pipeline=True.
"""

import ab.gpt.TuneNNGen_v2 as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_1.3b_instruct.json',
        run_iterative_pipeline=True,
        base_data_dir='curation_output/chat_data',
        cycles=28,
        models_per_cycle=20,
        resume_from_cycle=None,
    )


if __name__ == '__main__':
    main()
