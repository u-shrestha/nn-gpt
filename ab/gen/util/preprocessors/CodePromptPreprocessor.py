import json
import os

import ab.nn.api as nn_api

from overrides import override
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gen.util.preprocessors.PreprocessorBase import PreprocessorBase
from ab.gen.util.preprocessors._util import read_file


class CodePromptPreprocessor(PreprocessorBase):
    """
    Assumes the existance of accuracies.json and folder based dataset
    """
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, dataset_dir_path: str):
        super().__init__(max_len, tokenizer)
        self.dataset_dir_path = dataset_dir_path
        self.max_len = 0
        self.max_words = 0

    @override
    def get_raw_dataset(self) -> DataFrame:

        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        dataframe = nn_api.data()

        # TODO: Uncomment and correct all following code if necessary.

        #     DataFrame(columns=["instruction", "context", "response", "category", "text"])
        #
        # context_map = {} # json.loads(read_file(os.path.join(self.dataset_dir_path, "context.json")))
        #
        # for net_style in os.listdir(self.dataset_dir_path):
        #     net_path = os.path.join(self.dataset_dir_path, net_style)
        #     if os.path.isdir(net_path) and not "cifar" in net_path:
        #         code = read_file(os.path.join(net_path, "code.py"))
        #         acc_file = open(os.path.join(net_path, "accuracies.json"))
        #         accuracies = json.load(acc_file)
        #         acc_file.close()
        #         accuracies = [(acc[0], acc[1], acc[2]) for acc in accuracies.values()]
        #         context = context_map[net_style] if net_style in context_map else ""
        #         for instruction in json.loads(read_file(os.path.join(net_path, "prompts.json"))):
        #             for accuracy in accuracies:
        #                 prompt = instruction["prompt"]
        #                 prompt += " The model should achieve an accuracy of about " + str(accuracy[0]) + " after "
        #                 prompt += str(accuracy[1]) + ' epochs of training in the task of "' + str(accuracy[2]) + '".'
        #                 prompt += " Use PyTorch for the implementation."
        #                 prompt += ' Name the main class of the model "Net".'
        #                 prompt += " Provide only the code. Don't provide any explanation. "
        #                 prompt += "Remove any text from this reply."
        #                 category = instruction["classification"]
        #                 text = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "\n" + code}]
        #                 text = self.tokenizer.apply_chat_template(text, tokenize=False)
        #
        #                 next_line = {
        #                     "instruction": prompt,
        #                     "context": context,
        #                     "response": code,
        #                     "category": category,
        #                     "text": text
        #                 }
        #                 dataframe.loc[len(dataframe.index)] = list(next_line.values())
        #
        #                 if len(code) > self.max_len:
        #                     self.max_len += len(code)
        #
        #                 word_count = len(code.split(" "))
        #                 if word_count > self.max_words:
        #                     self.max_words = word_count
        return dataframe
