# Unsloth conditional import
try:
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    
import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='gpt_oss_20b.json')


if __name__ == '__main__':
    main()
