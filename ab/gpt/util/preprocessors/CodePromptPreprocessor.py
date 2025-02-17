import ab.nn as lemur

from overrides import override
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.preprocessors.PreprocessorBase import PreprocessorBase


class CodePromptPreprocessor(PreprocessorBase):
    """
    Assumes the existance of accuracies.json and folder based dataset
    """
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase):
        super().__init__(max_len, tokenizer)

    @override
    def get_raw_dataset(self) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        lemur_df = lemur.data()
        dataframe = DataFrame(columns=["instruction", "context", "response", "category", "text"])

        for idx, lemur_row in lemur_df.iterrows():
            inst = (
                f"Create a neural network structure in PyTorch for {lemur_row['task']} based on the {lemur_row['nn']} architecture,"
                f"that would achieve an accuracy of {lemur_row['accuracy']} after {lemur_row['epoch']} epochs of training"
                f"on the {lemur_row['dataset']} dataset. Name the main class 'Net'."
            )
            response = "```\n" + str(lemur_row['nn_code']) + "\n```"
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": response}
                ], tokenize=False
            )

            dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        return dataframe
