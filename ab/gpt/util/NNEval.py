import os
import ab.nn.api as api
from ab.nn.util.Util import read_py_file_as_string, uuid4
import ab.nn.api as nn_dataset


class NNEval:
    def __init__(self, model_source_package: str, task='img-classification', dataset='cifar-10', metric='acc', prm=None, save_to_db=False, prefix=None, save_path=None):
        """
        Evaluates a given model on a specified dataset for classification
        :param model_source_package: The package name of the model to evaluate
        :param task: The task to evaluate the model on
        :param dataset: The dataset to evaluate the model on
        :param metric: The metric to evaluate the model on
        :param prm: The parameters to evaluate the model on
        :param save_to_db: Whether to save the results to the database
        """
        if prm is None:
            prm = {'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9,
                   'transform': 'norm_256_flip', 'epoch': 1}
        self.model_package = model_source_package
        self.task = task
        self.dataset = dataset
        self.metric = metric
        self.prm = prm
        self.save_to_db = save_to_db
        self.prefix = prefix
        self.save_path = save_path

    def evaluate(self, nn_file):
        os.listdir(self.model_package)
        code = read_py_file_as_string(nn_file)
        ids_list = nn_dataset.data()["nn_id"].unique().tolist()
        new_checksum = uuid4(code)
        if new_checksum not in ids_list:
            res = api.check_nn(code, self.task, self.dataset, self.metric, self.prm, self.save_to_db, self.prefix, self.save_path)
            return res
        else:
            print(f'NN already exists (checksum: {new_checksum}). Skipping API call.')
            return None

    def get_args(self):
        return {
            'model_package': self.model_package,
            'task': self.task,
            'dataset': self.dataset,
            'metric': self.metric,
            'prm': self.prm,
        }
