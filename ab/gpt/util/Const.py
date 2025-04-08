from ab.nn.util.Const import base_module, ab_root_path, out_dir

new_nn_file = 'new_nn.py'
gpt = 'gpt'

gpt_dir = ab_root_path / base_module / gpt
conf_dir = gpt_dir / 'conf'
nngpt_dir = out_dir / 'nngpt'
acgpt_dir = out_dir / 'acgpt'

llm_prefix = 'ABrain/'

small_llm = f'{llm_prefix}NNGPT-DeepSeek-Coder-1.3B-Instruct'


def model_dir(base):
    return base / 'llm'


def synth_dir(base):
    return base / 'synth_nn'


def tokenizer_dir(base):
    return base / 'tokenizer'


nngpt_model = out_dir
nngpt_upload = model_dir(nngpt_dir / 'upload')
llm_tokenizer_out = tokenizer_dir(nngpt_dir)


def llm_dir(base, name):
    return model_dir(base) / name


def llm_tokenizer_dir(base, name):
    return tokenizer_dir(base) / name


def epoch_dir(epoch):
    return llm_dir(nngpt_dir, 'epoch') / f"A{epoch}"
