from ab.nn.util.Const import base_module, ab_root_path, out_dir

gpt = 'gpt'

gpt_dir = ab_root_path / base_module / gpt
conf_dir = gpt_dir / 'conf'

llm_prefix = 'ABrain/'

small_llm = f'{llm_prefix}NNGPT-DeepSeek-Coder-1.3B-Instruct'


def model_dir(base):
    return base / 'Models'


def tokenizer_dir(base):
    return base / 'Tokenizers'


nngpt_out = model_dir(out_dir)
nngpt_upload = model_dir(out_dir / 'upload')
llm_tokenizer_out = tokenizer_dir(out_dir)


def llm_weights_dir(base, name):
    return model_dir(base) / name


def llm_tokenizer_dir(base, name):
    return tokenizer_dir(base) / name


def epoch_dir(epoch):
    return llm_weights_dir(out_dir, 'epochs') / f"A{epoch}"
