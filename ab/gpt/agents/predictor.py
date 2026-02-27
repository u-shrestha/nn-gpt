"""
Predictor Agent - Predicts final accuracy (optional).
Requires TuneAccPrediction to be trained separately.
"""
from typing import Dict, Any
from ab.gpt.agents.state import AgentState


def predictor_node(state: AgentState) -> Dict[str, Any]:
    """Predict final accuracy from early-epoch metrics."""
    print("📊 Predictor: Running...")
    try:
        from ab.gpt.util.TuneAccPrediction import predict_best_accuracy
        model_code = state.get('model_code')
        epoch_1_acc = state.get('epoch_1_accuracy')
        epoch_2_acc = state.get('epoch_2_accuracy')
        if model_code and epoch_1_acc is not None and epoch_2_acc is not None:
            pred_acc, pred_epoch = predict_best_accuracy(
                model_code,
                epoch_1_acc,
                epoch_2_acc,
            )
            return {
                "status": "success",
                "predicted_best_accuracy": pred_acc,
                "predicted_best_epoch": pred_epoch,
                "gpu_available": True,
                "next_action": "end",
            }
        return {
            "status": "partial_success",
            "error_message": "Predictor needs model_code and epoch accuracies",
            "gpu_available": True,
            "next_action": "end",
        }
    except ImportError:
        return {
            "status": "partial_success",
            "error_message": "TuneAccPrediction not available",
            "gpu_available": True,
            "next_action": "end",
        }
    except Exception as e:
        return {
            "status": "partial_success",
            "error_message": str(e),
            "gpu_available": True,
            "next_action": "end",
        }
