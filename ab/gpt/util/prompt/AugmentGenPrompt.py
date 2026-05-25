import json
import glob
import os
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
from overrides import override
from tqdm import tqdm

from ab.gpt.util.prompt.Prompt import Prompt
from ab.gpt.util.Const import trans_dir



AUGMENT_RES_DIR = trans_dir / 'augment' / 'result'


def _augment_to_str(augment) -> str:
    """Serialise an augment list to a compact JSON string for prompt insertion"""
    return json.dumps(augment, separators=(',', ':'))


def load_augment_data(result_dir: str, only_best_accuracy: bool = True) -> DataFrame:
    """
    Load augmentation configs and acc from a folder of json files
    """

    print(f"Loading augment data from: {result_dir}", flush=True)
    all_data = []

    for path in tqdm(glob.glob(os.path.join(result_dir, "*.json")), desc="Reading augment JSONs"):
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Skip entries that recorded an error / zero accuracy
            if data.get('accuracy', 0.0) <= 0.0:
                continue
            if 'augment' not in data:
                print(f"Warning: 'augment' key missing in {path}, skipping.")
                continue

            data['id_name'] = os.path.splitext(os.path.basename(path))[0]
            data['augment_str'] = _augment_to_str(data['augment'])
            all_data.append(data)

        except Exception as e:
            print(f"Warning: Could not load {path}: {e}", flush=True)

    if not all_data:
        print("Warning: No augment data found. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Keep only the highest-accuracy entry per unique augment config
    if only_best_accuracy and 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False).drop_duplicates('id_name')

    print(f"Loaded {len(df)} augment data points.", flush=True)
    return df


def shuffle_data(df: DataFrame) -> DataFrame:
    return df.sample(frac=1).reset_index(drop=True)


class AugmentGenPrompt(Prompt):
    """
    Builds fine-tuning prompts from augment result JSON files.
    """

    def __init__(
        self,
        max_len: int,
        tokenizer: PreTrainedTokenizerBase,
        prompts_path,
        result_dir=None,
    ):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path
        self.result_dir = str(result_dir or AUGMENT_RES_DIR)


    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        Build a DataFrame of (instruction, context, response, category, text)
        rows ready for fine-tuning.
        """
        
        with open(self.prompts_path) as f:
            prompt_dict = json.load(f)
        assert isinstance(prompt_dict, dict)

        print('Preparing augment data...', flush=True)
        data = load_augment_data(self.result_dir, only_best_accuracy)

        if data.empty:
            print("No augment data found; returning empty dataset.")
            return pd.DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])

        data = shuffle_data(data)
        print('Augment data ready.', flush=True)

        prompt_lists = []

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)

            prompt_template = '\n'.join(prompt_dict[key]['prompt'])
            with_addons = 'addon_list' in prompt_dict[key] and prompt_dict[key]['addon_list']
            addon_data = data  # same pool used for addon sampling

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break

                row_dict = row.to_dict()
                para_dict = {}

                # ── fill input fields ──────────────────────────────────────
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row_dict.get(it['value'])

                # ── sample addon row ───────────────────────────────────────
            
                if with_addons:
                    # Create a safe boolean mask instead of a fragile string query
                    mask = addon_data['id_name'] != row['id_name']

                    if prompt_dict[key].get('improve') and row.get('accuracy') is not None:
                        mask &= (addon_data['accuracy'] > row['accuracy'])

                    for filter_it in prompt_dict[key].get('no_repeat', []):
                        if filter_it in row_dict:
                            mask &= (addon_data[filter_it] != row_dict[filter_it])

                    filtered = addon_data[mask]

                    if filtered.empty:
                        continue  # no better augment to learn from

                    addon_row = shuffle_data(filtered).iloc[0].to_dict()
                    for it in prompt_dict[key]['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])

                # ── format prompt & response ───────────────────────────────
                try:
                    inst = prompt_template.format(**para_dict)
                except KeyError as e:
                    print(f"Warning: Missing key {e} in prompt. Skipping.")
                    continue

                if 'output' not in prompt_dict[key]:
                    print(f"Warning: 'output' missing in config key '{key}'. Skipping.")
                    continue

                output_template = '\n'.join(prompt_dict[key]['output'])
                try:
                    response = output_template.format(**para_dict)
                except KeyError as e:
                    print(f"Warning: Missing key {e} in output template. Skipping.")
                    continue

                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user',      'content': inst},
                        {'role': 'assistant', 'content': response},
                    ],
                    tokenize=False,
                )
                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Augment prompts successfully generated.', flush=True)
        return pd.concat(prompt_lists, ignore_index=True)