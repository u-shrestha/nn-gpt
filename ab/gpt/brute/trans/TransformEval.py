import os
import json
import shutil
import argparse
from pathlib import Path
import ab.nn.api as api

from ab.gpt.util.Const import nngpt_dir, epoch_dir, synth_dir, transformer_file, trans_dir


# DEFAULT CONFIGURATION

RESULT_DIR = trans_dir/'test'
TRANSFORM_DIR = trans_dir/'test'

TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
MODEL_FILTER = "ResNet"


# HYPERPARAMETERS
DEFAULTS = {
    'lr': 0.01,
    'batch': 64,
    'dropout': 0.2,
    'momentum': 0.9,
    'epoch': 1
}



def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Transforms")
    
    # Mode Selection
    parser.add_argument('--mode', type=str, choices=['GEN', 'FT'], required=True, 
                        help="GEN: evaluate the generated transform in a single folder, FT: evaluate transforms generated during fine tuning.")
    parser.add_argument('--epoch_num', type=int, default=None, 
                        help="Fine tuning epoch num.")
    
    # Paths
    #parser.add_argument('--input', type=Path, default=TRANSFORM_DIR, help="Base directory containing the transform folders."),
    #parser.add_argument('--output', type=Path, default=RESULT_DIR, help="Directory to save results.")

    parser.add_argument('--threshold', type=float, default=0.40, 
                        help="Accuracy threshold to save 'best' models. Set 0 to disable.")
    
    return parser.parse_args()


def get_best_model_from_db():
    """
    Get best ResNet model from DB for CIFAR-10
    """
    df = api.data(
        only_best_accuracy=True,
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        nn=MODEL_FILTER,
        max_rows=1
    )
    
    if not df.empty:
        best_model = df.iloc[0].to_dict()
        print(f"Found best model: {best_model['nn']} with accuracy: {best_model['accuracy']}")
        return best_model
    else:
        raise ValueError("No matching models found in database")



def get_candidates(mode, epoch_num):
    """
    Scan for transform files based on the mode
    """

    candidates = []
    
    if mode == 'GEN':
        # Look for .py files directly
      
        search_dir = TRANSFORM_DIR

        print(f"Scanning in: {search_dir}")
        for entry in search_dir.glob("*.py"):
            candidates.append({
                'name': entry.stem,             # Filename is the ID
                'transform_param': entry.stem,  # API expects filename
                'context_dir': search_dir,      # API looks here
                'source_file': entry
            })

    elif mode == 'FT':
        # # Iterate through all subdirs in the path
        base_input_dir = synth_dir(epoch_dir(epoch_num)) 
        
        if not base_input_dir.exists():
            raise ValueError(f"Directory {base_input_dir} does not exist")
            
        for entry in os.scandir(base_input_dir):
            if entry.is_dir():
                tr_file = Path(entry.path) / "tr.py" 
                if tr_file.exists():
                    candidates.append({
                        'name': entry.name,      # Dir name is the ID
                        'transform_param': 'tr', # Fixed param name
                        'context_dir': Path(entry.path), # API looks inside subdir
                        'source_file': tr_file
                    })
    
    return candidates




def run_eval(mode, epoch_num):
    
    # Get model and candidates
    best_model = get_best_model_from_db()
    candidates = get_candidates(args.mode, args.epoch_num)
    
    print(f"Found {len(candidates)} transforms to evaluate.")

    for i, cand in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Evaluating: {cand['name']}")
        
        # Parameters
        prm = DEFAULTS.copy()
        prm.update({
            'transform': cand['transform_param']
        })

        try:
            # Run evaluation
            result = api.check_nn(
                nn_code=best_model['nn_code'],
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                prm=prm,
                save_to_db=False,
                prefix=f"{best_model['nn']}_eval_{cand['name']}",
                save_path= cand['context_dir'],
                transform_dir=cand['context_dir']
            )

            if result:
                _, accuracy, time_metric, _ = result
                print(f"  Result: Acc={accuracy}, Time={time_metric}")

                # Save JSON result
                result_data = {
                    "accuracy": float(accuracy),
                    "batch": prm['batch'],
                    "duration": prm['duration'],
                    "lr": prm['lr'],
                    "momentum": prm['momentum'],
                    "transform": cand['name'],
                    "uid": prm['uid']
                
                }
                
                json_path = cand['context_dir']/ f"{cand['name']}.json"
                with open(json_path, 'w') as f:
                    json.dump(result_data, f, indent=2)

                if mode == 'FT':
                    if args.threshold > 0 and float(accuracy) >= args.threshold:
                        print(f"  [SUCCESS] Accuracy >= {args.threshold}")
                        # Copy .py file
                        shutil.copy(cand['source_file'], TRANSFORM_DIR/f"A{epoch_num}{cand['name']}.py")
                        # Copy .json file
                        shutil.copy(json_path, RESULT_DIR/f"A{epoch_num}{cand['name']}.json")

        except Exception as e:
            print(f"  Error: {e}")
            error_data = {"accuracy": 0.0, "transform": cand['name'], "error": str(e)}
            with open(cand['context_dir'] / f"{cand['name']}.json", 'w') as f:
                json.dump(error_data, f, indent=2)



if __name__ == "__main__":
    args = get_args()
    run_eval(args.mode, args.epoch_num)