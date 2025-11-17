import os
import json
import shutil
import ab.nn.api as api
from pathlib import Path  
from ab.gpt.util.Const import nngpt_dir, epoch_dir, synth_dir, transformer_file, trans_dir


# Configuration

TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
MODEL_FILTER = "ResNet"

# Default settings
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_MOMENTUM = 0.9

OUTPUT_DIR = trans_dir/'result-gen'
TRANSFORM_DIR = trans_dir/'out-gen'


def get_best_model_from_db():
    """Get best ResNet model from DB for CIFAR-10"""
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



def run_evaluations(epoch_num=None):
    """Run evaluations with all generated transforms"""
    
    best_model = get_best_model_from_db()
   
    # Get model info
    model_code = best_model['nn_code']
    model_name = best_model['nn'] 
    
    # Iterate through all subdirs in the path from the image

    base_search_dir = synth_dir(epoch_dir(epoch_num)) 
   
    
    if not base_search_dir.exists():
        raise ValueError(f"Base search directory {base_search_dir} does not exist")

    # Find all subdirectories 
    transform_dirs_to_run = []
    for entry in os.scandir(base_search_dir):
        if entry.is_dir():
            # Check if tr.py exists inside
            tr_file_path = Path(entry.path) / transformer_file 
            if tr_file_path.exists():
                # Add the B* directory path
                transform_dirs_to_run.append(Path(entry.path)) 
    
    if not transform_dirs_to_run:
        raise ValueError(f"No directories containing {transformer_file} found in {base_search_dir}")
    
    print(f"Found {len(transform_dirs_to_run)} transform directories to evaluate")
    
    counter = 1
    # Loop through the found B directories
    for b_dir_path in transform_dirs_to_run:
        
        # Use the directory name
        eval_name = b_dir_path.name 
        
        # The transform parameter for the API
        transform_param_name = Path(transformer_file).stem 
        
        current_transform_dir = b_dir_path

        original_transform_file = b_dir_path / transformer_file
        
        # Copy transformer file
        new_transform_file = TRANSFORM_DIR / f"A{epoch_num}{eval_name}.py" 
        
        shutil.copy(original_transform_file, new_transform_file)
        
        
        print(f"\n[{counter}/{len(transform_dirs_to_run)}] Evaluating transform: {eval_name} (from {current_transform_dir})")
        
        base_prm = {
            'lr': DEFAULT_LR,
            'batch': DEFAULT_BATCH_SIZE,
            'dropout': DEFAULT_DROPOUT ,
            'momentum': DEFAULT_MOMENTUM,
            'transform': 'tr',
            'epoch': 1
        }
        
        try: 
            # Run evaluation 
            result = api.check_nn(
                nn_code=model_code,  
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                prm=base_prm,
                save_to_db=False,
                prefix=f"{model_name}_eval_{eval_name}", # Use B* name
                save_path=OUTPUT_DIR,
                transform_dir=current_transform_dir # Pass the .../B0 directory
            )
            
            
            if result:
                result_model_name, accuracy, time_metric, acc_time_ratio = result
                
                print(f"  Result: Acc={accuracy}, Time={time_metric}, Acc/Time={acc_time_ratio}")

                result_data = {
                    "accuracy": float(accuracy),
                    "time metric": time_metric,
                    "batch": base_prm['batch'],
                    "duration": base_prm['duration'],
                    "lr": base_prm['lr'],
                    "momentum": base_prm['momentum'],
                    "transform": f"A{epoch_num}{eval_name}", 
                    "prm": json.dumps(base_prm),
                    "uid": base_prm['uid']
                }
                
                result_file_path = OUTPUT_DIR / f"A{epoch_num}{eval_name}.json" 
                with open(result_file_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                print(f"  Accuracy: {accuracy:.4f} - Saved to {result_file_path}")
                counter += 1
                

        except Exception as e:
            print(f"  Error evaluating {eval_name}: {str(e)}")
            error_data = {
                "accuracy": 0.0,
                "duration": 0,
                "time metric": 0.0,
                "batch": base_prm['batch'],
                "lr": base_prm['lr'],
                "momentum": base_prm['momentum'],
                "transform": eval_name,
                "prm": json.dumps(base_prm),
                "uid": "",
                "error": str(e)
            }
    
            result_file_path = os.path.join(OUTPUT_DIR, f"A{epoch_num}{eval_name}.json") 
            with open(result_file_path, 'w') as f:
                json.dump(error_data, f, indent=2)
            counter += 1


    print(f"\nEvaluation completed. Results saved to {OUTPUT_DIR}")
    return OUTPUT_DIR

if __name__ == "__main__":
    results_path = run_evaluations()