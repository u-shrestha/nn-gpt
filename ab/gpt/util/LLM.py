from ab.nn.util.Const import out_dir
from ab.gpt.util.Const import llm_dir, llm_tokenizer_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Util import exists

import torch.cuda
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)


class LLM:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None,
                 use_deepspeed=False,
                 base_path=out_dir,
                 context_length=None):
        # Load the tokenizer
        self.context_length = context_length
        tok_fl_nm = llm_tokenizer_dir(base_path, model_path)
        raw_fl_nm = llm_dir(base_path, model_path)
        if exists(tok_fl_nm):
            print("Loading Tokenizer from local files:", tok_fl_nm)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_fl_nm, token=access_token,
                quantization_config=quantization_config_4bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            print("Downloading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=access_token)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.tokenizer.save_pretrained(tok_fl_nm, access_token=access_token)
            print("Tokenizer saved to: ", tok_fl_nm)

        # Load the model
        if use_deepspeed:  # When using Deepspeed, device_map should not be given, for deepspeed automatically manages the device memory mapping
            if local_path and exists(local_path):
                print("Loading Model from local files:", "'" + local_path + "'")
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    max_memory={i: max_memory for i in range(torch.cuda.device_count())},
                    token=access_token)
            elif exists(raw_fl_nm):
                print("Loading Model from local files:", raw_fl_nm)
                self.model = AutoModelForCausalLM.from_pretrained(raw_fl_nm, max_memory={i: max_memory for i in range(torch.cuda.device_count())},
                                                                  token=access_token)
            else:
                print("Downloading Model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    max_memory={i: max_memory for i in range(torch.cuda.device_count())},
                    token=access_token)
                self.model.save_pretrained(raw_fl_nm, access_token=access_token)
                print("Model saved to: ", raw_fl_nm)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path if exists(local_path) else raw_fl_nm if exists(raw_fl_nm) else model_path,
                trust_remote_code=True,
                device_map="auto",
                max_memory={i: max_memory for i in range(torch.cuda.device_count())},
                token=access_token,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
            )
            if exists(local_path):
                print("Loading Model from local files:", "'" + local_path + "'")
            elif exists(raw_fl_nm):
                print(f"Loading Raw Model from local files: '{raw_fl_nm}'")
            else:
                self.model.save_pretrained(raw_fl_nm, access_token=access_token)
                print("Model saved to: ", raw_fl_nm)

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

    def get_max_length(self) -> int:
        if self.context_length:
            return self.context_length
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        if not max_length:
            max_length = 4096
        return max_length
