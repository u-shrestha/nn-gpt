import gc
import json
import os
import shutil
from pathlib import Path
import torch


import ab.nn.api as api
from ab.gpt.util.Const import epoch_dir, synth_dir, trans_dir



# --- CONFIGURATION ---
FT_MODE = True  # Set to 'False' for evaluation of local transform files in a single folder
RESULT_DIR     = trans_dir / 'augment' / 'gen_result'  # Directory to save final results (only used in FT_MODE)
LOCAL_EVAL_DIR = trans_dir / 'augment' / 'result'     # Directory for standalone local evaluation

TASK           = "img-classification"
DATASET        = "cifar-10"
METRIC         = "acc"
MODEL_FILTER   = "ResNet"
THRESHOLD      = 0.40  # minimum accuracy to keep a generated config
AUGMENT_FILE   = "aug.json" # filename the generation step writes

# --- HYPERPARAMETERS ---
DEFAULTS = {
    'lr':        0.01,
    'batch':     16,
    'dropout':   0.2,
    'momentum':  0.9,
    'epoch':     1,
    'transform': "norm", 
}


def get_best_model_from_db():
    """Return the best ResNet entry for the configured task/dataset from the DB."""
    df = api.data(
        only_best_accuracy=True,
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        nn=MODEL_FILTER,
        max_rows=1,
    )
    if not df.empty:
        best = df.iloc[0].to_dict()
        print(f"Best model: {best['nn']}  accuracy={best['accuracy']}")
        return best
    raise ValueError("No matching model found in the database.")


def get_candidates(epoch_num=None, FT_MODE=False):
    """
    Scan for augment configuration files based on the mode.
    Handles both single files and batch dictionary files.
    """
    candidates = []

    if not FT_MODE:
        # Local Mode: Scan a dedicated test folder for json configs
        print(f"Scanning local directory for configs: {LOCAL_EVAL_DIR}")
        LOCAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        
        for entry in LOCAL_EVAL_DIR.glob("*.json"):
            # Skip files that are already evaluation results
            if entry.name.endswith("_result.json") or entry.name.endswith("_error.json"):
                continue
                
            try:
                with open(entry, 'r') as f:
                    data = json.load(f)
                
                # Check if this is a "batch file"
                if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                    print(f"  -> Found batch config file: {entry.name}")
                    for key, augment_list in data.items():
                        candidates.append({
                            'name': f"{entry.stem}_config_{key}", # e.g., augment_configs_seg1_n10_config_1
                            'augment_data': augment_list,         # Pass the list directly in memory
                            'context_dir': LOCAL_EVAL_DIR,      
                        })
                else:
                    
                    # Handle cases where the list is wrapped in {"augment": [...]} or is just a raw list
                    aug_list = data.get('augment', data) if isinstance(data, dict) else data
                    candidates.append({
                        'name': entry.stem,          
                        'augment_data': aug_list,               
                        'context_dir': LOCAL_EVAL_DIR,      
                    })
            except Exception as e:
                print(f"Error reading {entry}: {e}")

    else:
        # FT Mode: Scan the synthesis directory for B{idx}/aug.json files
        base_dir = synth_dir(epoch_dir(epoch_num))
        if not base_dir.exists():
            raise ValueError(f"Synthesis directory does not exist: {base_dir}")

        print(f"Scanning synthesis directory: {base_dir}")
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            aug_file = Path(entry.path) / AUGMENT_FILE
            if aug_file.exists():
                try:
                    with open(aug_file, 'r') as f:
                        augment_data = json.load(f)
                    candidates.append({
                        'name': entry.name,
                        'augment_data': augment_data,
                        'context_dir': Path(entry.path),
                    })
                except Exception as e:
                    print(f"Error reading {aug_file}: {e}")

    return candidates


def run_eval(epoch_num=None, FT_MODE=False):
    """
    Evaluate all augmentation configs for the given epoch or local folder.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    best_model = get_best_model_from_db()
    candidates = get_candidates(epoch_num, FT_MODE)
    print(f"Found {len(candidates)} augment candidates to evaluate (FT_MODE={FT_MODE}).")

    for i, cand in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Evaluating: {cand['name']}")

        augment = cand['augment_data']
        if not isinstance(augment, list):
            print(f"  Skipping: Data is not a list -> {augment}")
            continue

        context_dir = cand['context_dir']

        # Determine output filename (same path regardless of mode — dead branch removed)
        local_json = context_dir / f"{cand['name']}.json"

        if local_json.exists():
            print(f"  Skipping: Result already exists ({local_json.name})")
            continue

        prm = DEFAULTS.copy()
        prm['augment'] = augment

        try:
            result = api.check_nn(
                nn_code=best_model['nn_code'],
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                prm=prm,
                save_to_db=False,
                prefix=f"{best_model['nn']}_augeval_{cand['name']}",
                save_path=context_dir,
            )

            if result:
                _, accuracy, time_metric, _ = result
                print(f"  Acc={accuracy}  Time={time_metric}")

                result_data = {
                    "dataset":   DATASET,
                    "model":     MODEL_FILTER,
                    "accuracy":  float(accuracy),
                    "batch":     prm['batch'],
                    "duration":  prm.get('duration', float(time_metric)), 
                    "lr":        prm['lr'],
                    "momentum":  prm['momentum'],
                    "transform": prm['transform'],
                    "augment":   augment,
                    "uid":       prm.get('uid', ""),
                }

                with open(local_json, 'w') as f:
                    json.dump(result_data, f, indent=2)

                if FT_MODE:
                    if THRESHOLD > 0 and float(accuracy) >= THRESHOLD:
                        print(f"  [KEEP] accuracy {accuracy} >= threshold {THRESHOLD}")
                        dest_json = RESULT_DIR / f"E{epoch_num}_{cand['name']}.json"
                        shutil.copy(local_json, dest_json)
                        print(f"  Saved to {dest_json}")

        except Exception as e:
            print(f"  Error during evaluation: {e}")
            error_data = {
                "accuracy": 0.0,
                "augment":  augment,
                "name":     cand['name'],
                "error":    str(e),
            }
            error_json = context_dir / f"{cand['name']}_error.json"
            with open(error_json, 'w') as f:
                json.dump(error_data, f, indent=2)

        finally:
            # Always release GPU memory after each candidate — prevents cuDNN
            _release_gpu_memory()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=None, help="Epoch number to evaluate.")
    parser.add_argument('--ft_mode', action='store_true', default=False, help="Enable fine-tuning mode to scan synthesis directories.")
    args = parser.parse_args()
    
    run_eval(epoch_num=args.epoch, FT_MODE=args.ft_mode)