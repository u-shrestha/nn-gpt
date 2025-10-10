import json

import ab.nn as lemur
from overrides import override
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from tqdm import tqdm


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        prompt_lists = []

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
            assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            # Get nn-dataset codes
            print('Preparing Data...', flush=True)
            data = lemur.data(only_best_accuracy=only_best_accuracy, task=prompt_dict[key]['task'])
            data = shuffle_data(data)
            print('Data acquisition complete', flush=True)
            with_addons = prompt_dict[key]['addon_list']

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]

                if with_addons:
                    param_names = prompt_dict[key]['keep_same']
                    params = {k: row[k] for k in param_names}
                    addon_task = prompt_dict[key]['addon_task']
                    if addon_task:
                        params['task'] = addon_task
                    addon_data = lemur.data(only_best_accuracy=only_best_accuracy, **params)

                    filter_q = f"nn!='{row['nn']}'"
                    if 'same_pref' in prompt_dict[key] and prompt_dict[key]['same_pref'] == True:
                        filter_q += f"&nn.str.split('-',n=1).str[0]=='{row['nn'].split('-')[0]}'"  # Default addon filter in this case should be models with same prefix

                    if 'improve' in prompt_dict[key] and prompt_dict[key]['improve'] == True:
                        filter_q += f"&accuracy>{row['accuracy']}"  # model result with higher accuracy

                    ## Apply non-repeat filter if exists:
                    if 'no_repeat' in prompt_dict[key]:  # Compact test_prompt.json, items in prm is not supported
                        for filter_it in prompt_dict[key]['no_repeat']:
                            if isinstance(row[filter_it], str):
                                filter_q += f"&{filter_it}!='{row[filter_it]}'"
                            else:
                                filter_q += f"&{filter_it}!={row[filter_it]}"
                    filtered_addon_data = addon_data.query(filter_q)
                    del addon_data

                    if len(filtered_addon_data) > 0:
                        shuffled = shuffle_data(filtered_addon_data)
                        addon_row = shuffled.sample(n=1).iloc[0]
                        del filtered_addon_data, shuffled
                    else:
                        continue  # The case no result matches requirement
                    for it in prompt_dict[key]['addon_list']:
                        para_dict[it['para']] = addon_row[it['value']]

                inst = prompt.format(**para_dict)
                # Having the same name prefix before '-'
                output = '\n'.join(prompt_dict[key]['output'])
                response = output.format(**para_dict)
                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False
                )

                # print(f"Prompt: {inst}")
                # print(f"Output: {response}")

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data
        return pd.concat(prompt_lists, ignore_index=True)
