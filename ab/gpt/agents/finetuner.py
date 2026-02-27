"""
Finetuner Agent - Performs LoRA finetuning (same logic as classic tune() loop).
Ensures agent path incorporates all improvements from tune().
"""
from typing import Dict, Any
from pathlib import Path

from ab.gpt.agents.state import AgentState
from ab.nn.util.Util import release_memory


# Import transform paths from Tune's scope - will be passed via state
def finetuner_node(state: AgentState) -> Dict[str, Any]:
    """
    Run one epoch of finetuning - identical to classic tune() loop body.
    Uses model, lora_tuner, etc. from state (loaded by tune()).
    """
    print("📚 Finetuner: Running LoRA finetune (same as classic tune)...")
    try:
        model = state.get('model')
        tokenizer = state.get('tokenizer')
        model_loader = state.get('model_loader')
        lora_tuner = state.get('lora_tuner')
        train_config_path = state.get('train_config_path')
        base_model_name = state.get('base_model_name')
        only_best_accuracy = state.get('only_best_accuracy', True)
        max_prompts = state.get('max_prompts')
        max_new_tokens = state.get('max_new_tokens', 16 * 1024)
        context_length = state.get('context_length')
        use_unsloth = state.get('use_unsloth', False)
        unsloth_max_input_length = state.get('unsloth_max_input_length')
        trans_mode = state.get('trans_mode', False)
        current_epoch = state.get('current_epoch', 0)

        if model is None or lora_tuner is None:
            return {
                "status": "error",
                "error_message": "model or lora_tuner missing - finetuner must run inside tune()",
                "gpu_available": True,
                "next_action": "end",
            }

        from ab.gpt.util.Const import epoch_dir
        out_path = Path(epoch_dir(current_epoch))

        if trans_mode:
            from ab.gpt.util.Const import trans_dir
            from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt
            TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
            TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'
            data_processor = TransformGenPrompt(
                context_length if context_length else model_loader.get_max_length(),
                tokenizer,
                train_config_path,
                TRANSFORM_OUT_DIR,
                TRANSFORM_RES_DIR,
            )
        else:
            from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
            max_len = unsloth_max_input_length if use_unsloth and unsloth_max_input_length else (
                context_length if context_length else model_loader.get_max_length()
            )
            data_processor = NNGenPrompt(max_len, tokenizer, train_config_path)

        dataset = data_processor.get_dataset(only_best_accuracy, max_prompts=max_prompts, max_new_tokens=max_new_tokens)
        print(f'Dataset length: {len(dataset)}')
        model.train()
        trained_model = lora_tuner.train(dataset, tokenizer, out_path / base_model_name)
        del dataset
        release_memory()

        from ab.gpt.util.Chatbot import ChatBot
        temperature = state.get('temperature', 0.8)
        top_k = state.get('top_k', 70)
        top_p = state.get('top_p', 0.9)
        chat_bot = ChatBot(trained_model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)

        return {
            "model": trained_model,
            "chat_bot": chat_bot,
            "current_epoch": current_epoch + 1,
            "gpu_available": True,
            "next_action": "continue",
        }
    except Exception as e:
        print(f"❌ Finetuner: Error - {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": str(e),
            "gpu_available": True,
            "next_action": "end",
        }
