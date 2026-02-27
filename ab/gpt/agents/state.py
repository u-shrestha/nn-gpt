"""
Shared state for LangGraph agents.
All fields are optional - agents add/update as needed.
"""
from typing import TypedDict, Any, Optional


class AgentState(TypedDict, total=False):
    experiment_id: str
    base_out_dir: str
    num_train_epochs: int
    nn_train_epochs: int
    nn_gen_conf_id: str
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    save_llm_output: bool
    llm_conf: str
    nn_gen_conf: str
    use_predictor: bool
    gpu_available: bool
    next_action: str
    status: str
    chat_bot: Any
    prompt_dict: dict
    conf_keys: tuple
    test_nn: int
    unsloth_max_input_length: Optional[int]
    prompt_batch: int
    train_config_path: str
    base_model_name: str
    only_best_accuracy: bool
    trans_mode: bool
    model: Any
    tokenizer: Any
    model_loader: Any
    lora_tuner: Any
    model_code: str
    nn_name: str
    accuracy: float
    predicted_best_accuracy: float
    predicted_best_epoch: int
    epoch_1_accuracy: float
    epoch_2_accuracy: float
    error_message: str
    eval_results: Any
    eval_args: Any
    cli_args: Any
