"""
Manager Agent - Routes workflow between Generator, Finetuner, and Predictor.
Orchestrates the full tune() loop: generate -> finetune -> ... for each epoch.
"""
from typing import Dict, Any
from ab.gpt.agents.state import AgentState


def manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Route: generate -> generator, finetune -> finetuner, predict -> predictor, end -> done.
    Implements same multi-epoch loop as classic tune(): generate then finetune, repeat.
    """
    print("🎛️ Manager: Coordinating workflow...")

    if state.get('status') == 'error':
        print("❌ Manager: Error detected → Ending")
        return {"next_action": "end", "gpu_available": True}

    next_action = state.get('next_action', 'generate')
    current_epoch = state.get('current_epoch', 0)
    llm_tune_epochs = state.get('llm_tune_epochs', 3)
    skip_epoch = state.get('skip_epoch', 1)
    use_predictor = state.get('use_predictor', False)
    gpu_available = state.get('gpu_available', True)

    # --- From finetuner: continue to next epoch or end ---
    if next_action == 'continue':
        if current_epoch >= llm_tune_epochs:
            # All epochs done - optional predictor or end
            if use_predictor:
                has_epoch_data = (
                    state.get('epoch_1_accuracy') is not None
                    and state.get('epoch_2_accuracy') is not None
                )
                if has_epoch_data:
                    print("✅ Manager → Predictor (all epochs done)")
                    return {"next_action": "predict", "gpu_available": False}
            print("✅ Manager → End (all epochs done)")
            return {"next_action": "end", "gpu_available": True}
        # Next epoch
        if current_epoch < skip_epoch:
            print(f"✅ Manager → Finetuner (epoch {current_epoch} < skip, skip generate)")
            return {"next_action": "finetune", "current_epoch": current_epoch}
        print(f"✅ Manager → Generator (epoch {current_epoch})")
        return {"next_action": "generate", "current_epoch": current_epoch}

    # --- From generator: always go to finetune ---
    if next_action == 'finetune':
        print("✅ Manager → Finetuner")
        return {"next_action": "finetune"}

    # --- Predictor path (optional) ---
    has_model = (
        state.get('model_code') is not None
        and str(state.get('model_code', '')).strip() != ''
        and state.get('status') != 'error'
    )
    has_prediction = (
        state.get('predicted_best_accuracy') is not None
        and state.get('predicted_best_epoch') is not None
    )
    has_epoch_data = (
        state.get('epoch_1_accuracy') is not None
        and state.get('epoch_2_accuracy') is not None
    )
    predictor_error = str(state.get('error_message', ''))
    predictor_tried = (
        state.get('status') == 'partial_success'
        and predictor_error
        and ('model not available' in predictor_error.lower()
             or 'TuneAccPrediction' in predictor_error
             or 'not yet implemented' in predictor_error.lower())
    )

    if next_action == 'predict':
        if predictor_tried:
            print("⚠️ Manager: Predictor already tried → Ending")
            return {"next_action": "end", "gpu_available": True}
        if not has_epoch_data and use_predictor:
            print("⚠️ Manager: Predictor needs epoch data → Ending")
            return {"next_action": "end", "gpu_available": True}
        if has_model and use_predictor and not has_prediction and gpu_available:
            print("✅ Manager → Predictor")
            return {"next_action": "predict", "gpu_available": False}

    # --- Initial or generate: start of epoch ---
    if next_action == 'generate' or next_action == 'end':
        if next_action == 'end':
            print("✅ Manager → End")
            return {"next_action": "end", "gpu_available": True}
        if current_epoch < skip_epoch:
            print(f"✅ Manager → Finetuner (epoch {current_epoch} < skip)")
            return {"next_action": "finetune", "current_epoch": current_epoch}
        print(f"✅ Manager → Generator (epoch {current_epoch})")
        return {"next_action": "generate", "current_epoch": current_epoch}

    print("✅ Manager → End")
    return {"next_action": "end", "gpu_available": True}
