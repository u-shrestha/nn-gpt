import os
import os.path
from ab.nn.util.Const import out_dir

import torch.cuda
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)


class ModelLoader:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None,
                 use_deepspeed=False,
                 base_path=out_dir):
        self.model_path = model_path
        self.bnb_config = bnb_config
        self.max_memory = max_memory
        self.access_token = access_token
        self.tokenizer = None
        self.model = None
        self.local_path = local_path
        self.use_deepspeed = use_deepspeed
        self.base_path = base_path
        self.initialize()

    def initialize(self):
        # Load the tokenizer
        tok_fl_nm = self.base_path / 'Tokenizers' / self.model_path
        raw_fl_nm = self.base_path / "Models" / (self.model_path + "_raw")
        if os.path.exists(tok_fl_nm):
            print("Loading Tokenizer from local files:", tok_fl_nm)
            self.tokenizer = AutoTokenizer.from_pretrained(tok_fl_nm, token=self.access_token)
        else:
            print("Downloading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, token=self.access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.tokenizer.save_pretrained(tok_fl_nm, access_token=self.access_token)
            print("Tokenizer saved to: ", tok_fl_nm)

        # Load the model
        if self.use_deepspeed: # When using Deepspeed, device_map should not be given, for deepspeed automatically manages the device memory mapping
            if self.local_path and os.path.exists(self.local_path):
                print("Loading Model from local files:", "'" + self.local_path + "'")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_path,
                    max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
            elif os.path.exists(raw_fl_nm):
                print("Loading Model from local files:", raw_fl_nm)
                self.model = AutoModelForCausalLM.from_pretrained(raw_fl_nm, max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
            else:
                print("Downloading Model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
                self.model.save_pretrained(raw_fl_nm, access_token=self.access_token)
                print("Model saved to: ", raw_fl_nm)
        else:
            if self.local_path and os.path.exists(self.local_path):
                print("Loading Model from local files:", "'" + self.local_path + "'")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_path,
                    device_map="auto",
                    max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
            elif os.path.exists(raw_fl_nm):
                print("Loading Model from local files:", raw_fl_nm)
                self.model = AutoModelForCausalLM.from_pretrained(raw_fl_nm,
                    device_map="auto",
                    max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
            else:
                print("Downloading Model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    # Seems a conversion after downloading the model.
                    # This will cause error when running the script even not enabling deep speed.
                    # quantization_config=self.bnb_config, 
                    device_map="auto",
                    max_memory={i: self.max_memory for i in range(torch.cuda.device_count())},
                    token=self.access_token)
                self.model.save_pretrained(raw_fl_nm, access_token=self.access_token)
                print("Model saved to: ", raw_fl_nm)

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

    def get_max_length(self) -> int:
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        if not max_length:
            max_length = 4096
        return max_length
