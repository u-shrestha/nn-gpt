import os
import os.path

import torch.cuda
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase, PreTrainedModel

)


class ModelLoader:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None
                 ):
        self.model_path = model_path
        self.bnb_config = bnb_config
        self.max_memory = max_memory
        self.access_token = access_token
        self.tokenizer = None
        self.model = None
        self.local_path = local_path
        self.initialize()

    def initialize(self):
        # Load the tokenizer
        if os.path.exists("./Tokenizers/" + self.model_path):
            print("Loading Tokenizer from local files:", '"./Tokenizers/' + self.model_path + '"')
            self.tokenizer = AutoTokenizer.from_pretrained(
                "./Tokenizers/" + self.model_path, token=self.access_token
            )
        else:
            print("Downloading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=self.access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.tokenizer.save_pretrained("./Tokenizers/" + self.model_path, access_token=self.access_token)
            print("Tokenizer saved to: ", "./Tokenizers/" + self.model_path)

        # Load the model
        if self.local_path and os.path.exists(self.local_path):
            print("Loading Model from local files:", "'" + self.local_path + "'")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token
            )
        elif os.path.exists("./Models/" + self.model_path + "_raw"):
            print("Loading Model from local files:", '"./Models' + self.model_path + '_raw"')
            self.model = AutoModelForCausalLM.from_pretrained(
                "./Models/" + self.model_path + "_raw",
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token
            )
        else:
            print("Downloading Model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=self.bnb_config,
                device_map="auto",
                max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                token=self.access_token
            )
            self.model.save_pretrained("./Models/" + self.model_path + "_raw", access_token=self.access_token)
            print("Model saved to: ", "./Models/" + self.model_path + "_raw")

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def get_max_length(self) -> int:
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        if not max_length:
            max_length = 1024
        return max_length
