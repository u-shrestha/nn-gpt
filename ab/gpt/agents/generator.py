"""
Generator Agent - Uses pre-loaded chat_bot from tune() state.
Calls nn_gen() only. No load_llm_and_chatbot, load_prompt_config, or read_eval_info.
"""
from typing import Dict, Any
from pathlib import Path
import json

from ab.gpt.agents.state import AgentState
from ab.gpt.util.Const import epoch_dir, synth_dir
from ab.gpt.util.Tune import nn_gen


def _read_eval_info(model_dir: Path) -> dict:
    """Read eval_info.json from model directory. No import from Tune."""
    eval_file = model_dir / 'eval_info.json'
    if not eval_file.exists():
        return {}
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Generator: Error reading eval_info.json: {e}")
        return {}


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generator agent - uses chat_bot from state (loaded by tune()).
    Calls nn_gen() with pre-loaded resources. Truly integrated.
    """
    print("🤖 Generator: Starting generation (using tune()'s chat_bot)...")

    try:
        chat_bot = state.get('chat_bot')
        if chat_bot is None:
            return {
                "status": "error",
                "error_message": "chat_bot not in state - agent must run inside tune()",
                "gpu_available": True,
                "next_action": "end",
            }

        prompt_dict = state.get('prompt_dict')
        conf_keys = state.get('conf_keys')
        test_nn = state.get('test_nn', 1)
        nn_train_epochs = state.get('nn_train_epochs', 1)
        max_new_tokens = state.get('max_new_tokens', 16 * 1024)
        save_llm_output = state.get('save_llm_output', True)
        nn_name_prefix = state.get('experiment_id') or state.get('nn_gen_conf_id') or 'exp_default'
        unsloth_max_input_length = state.get('unsloth_max_input_length')
        prompt_batch = state.get('prompt_batch', 1)
        epoch = state.get('current_epoch', 0)

        if prompt_dict is None or conf_keys is None:
            return {
                "status": "error",
                "error_message": "prompt_dict or conf_keys missing from state",
                "gpu_available": True,
                "next_action": "end",
            }

        out_path = epoch_dir(epoch)

        nn_gen(
            epoch=epoch,
            out_path=out_path,
            chat_bot=chat_bot,
            conf_keys=conf_keys,
            nn_train_epochs=nn_train_epochs,
            prompt_dict=prompt_dict,
            test_nn=test_nn,
            max_new_tokens=max_new_tokens,
            save_llm_output=save_llm_output,
            nn_name_prefix=nn_name_prefix,
            unsloth_max_input_length=unsloth_max_input_length,
            prompt_batch=prompt_batch,
        )

        print("✅ Generator: nn_gen() completed!")

        models_dir = synth_dir(out_path)
        output_dir = models_dir / 'B0'
        if not output_dir.exists():
            return {
                "status": "partial_success",
                "error_message": f"Output directory not found: {output_dir}",
                "model_code": "",
                "gpu_available": True,
                "next_action": "finetune",
            }

        model_file = output_dir / 'new_nn.py'
        model_code = None
        if model_file.exists():
            try:
                model_code = model_file.read_text(encoding='utf-8')
                if model_code and model_code.strip():
                    print(f"📄 Generator: Model code read from new_nn.py ({len(model_code)} chars)")
            except Exception as e:
                print(f"⚠️ Generator: Error reading new_nn.py: {e}")

        eval_info = _read_eval_info(output_dir)
        eval_results = eval_info.get('eval_results', [])
        eval_args = eval_info.get('eval_args', {})
        cli_args = eval_info.get('cli_args', {})

        nn_name = eval_results[0] if len(eval_results) > 0 else None
        accuracy = eval_results[1] if len(eval_results) > 1 else None
        epoch_1_accuracy = None
        epoch_2_accuracy = None

        try:
            epoch_data = eval_info.get('epoch_accuracies', {})
            epoch_1_accuracy = epoch_data.get(1) or epoch_data.get('1')
            epoch_2_accuracy = epoch_data.get(2) or epoch_data.get('2')
        except Exception:
            pass

        has_model = model_code and model_code.strip()
        has_metrics = accuracy is not None or len(eval_results) > 1
        status = 'success' if (has_model and has_metrics) else ('partial_success' if has_model else 'error')
        error_message = None if status == 'success' else (
            'Model generated but no metrics' if has_model else 'Model code could not be read'
        )

        return {
            'model_code': model_code or '',
            'nn_name': nn_name,
            'accuracy': accuracy,
            'epoch_1_accuracy': epoch_1_accuracy,
            'epoch_2_accuracy': epoch_2_accuracy,
            'eval_results': eval_results if eval_results else None,
            'eval_args': eval_args if eval_args else None,
            'cli_args': cli_args if cli_args else None,
            'status': status,
            'gpu_available': True,
            'error_message': error_message,
            'next_action': 'finetune',
        }

    except Exception as e:
        print(f"❌ Generator: Error - {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": str(e),
            "gpu_available": True,
            "next_action": "end",
        }
