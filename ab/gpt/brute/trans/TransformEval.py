import os
import json
import ab.nn.api as api



# Configuration
OUTPUT_DIR = "ab/gpt/brute/trans/results"
TRANSFORM_DIR = "ab/gpt/brute/trans/transform_files"
TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
MODEL_FILTER = "ResNet"

# Default settings
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_MOMENTUM = 0.9



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



def run_evaluations():
    """Run evaluations with all generated transforms"""
    
    best_model = get_best_model_from_db()
   
    # Get model info
    model_code = best_model['nn_code']
    model_name = best_model['nn'] 
    
    
    # Create parent folder with specified structure
    parent_folder_name = f"{TASK}_{DATASET}_{METRIC}_{model_name}"
    parent_folder_path = os.path.join(OUTPUT_DIR, parent_folder_name)
    os.makedirs(parent_folder_path, exist_ok=True)

    
    # Iterate through all .py files in transform directory
    if not os.path.exists(TRANSFORM_DIR):
        raise ValueError(f"Transform directory {TRANSFORM_DIR} does not exist")
    
    transform_files = [f for f in os.listdir(TRANSFORM_DIR) if f.endswith('.py')]
    
    if not transform_files:
        raise ValueError(f"No .py files found in {TRANSFORM_DIR}")
    
    print(f"Found {len(transform_files)} transform files to evaluate")
    
    counter = 1
    for transform_file in transform_files:
        # Extract transform name
        transform_name = os.path.splitext(transform_file)[0]
        print(f"\n[{counter}/{len(transform_files)}] Evaluating transform: {transform_name}")
        
        base_prm = {
            'lr': DEFAULT_LR,
            'batch': DEFAULT_BATCH_SIZE,
            'dropout': DEFAULT_DROPOUT ,
            'momentum': DEFAULT_MOMENTUM,
            'transform': transform_name,
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
                prefix=f"{model_name}_eval_{transform_name}",
                save_path=parent_folder_path,
                transform_dir=TRANSFORM_DIR
                
            )
            
            
            if result:
                # Unpack results
                result_model_name, accuracy, time_metric, _ = result
                
                # Create result in specified format
                result_data = {
                    "accuracy": float(accuracy),
                    "batch": base_prm['batch'],
                    "duration": base_prm['duration'],
                    "lr": base_prm['lr'],
                    "momentum": base_prm['momentum'],
                    "transform": transform_name,
                    "uid": base_prm['uid']
                }
                
                # Save to individual JSON file
                result_file_path = os.path.join(parent_folder_path, f"{counter}.json")
                with open(result_file_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                print(f"  Accuracy: {accuracy:.4f} - Saved to {result_file_path}")
                counter += 1

        except Exception as e:
            print(f"  Error evaluating {transform_name}: {str(e)}")
            # Save error info in same format
            error_data = {
                "accuracy": 0.0,
                "batch": base_prm['batch'],
                "duration": 0,
                "lr": base_prm['lr'],
                "momentum": base_prm['momentum'],
                "transform": transform_name,
                "uid": "",
                "error": str(e)
            }
            result_file_path = os.path.join(parent_folder_path, f"{counter}.json")
            with open(result_file_path, 'w') as f:
                json.dump(error_data, f, indent=2)
            counter += 1

    print(f"\nEvaluation completed. Results saved to {parent_folder_path}")
    return parent_folder_path

if __name__ == "__main__":
    results_path = run_evaluations()