from ab.nn.util.Const import base_module, ab_root_path, out_dir

gpt = 'gpt'

gpt_dir = ab_root_path / base_module / gpt
conf_dir = gpt_dir / 'conf'


def epoch_dir(epoch):
    return out_dir / 'Models' / 'epochs' / f"A{epoch}"
