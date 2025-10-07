import argparse, json, sys, os, traceback
from pathlib import Path
import pandas as pd
from ab.nn.util.Util import release_memory, uuid4, read_py_file_as_string

from ab.gpt.util.Const import epoch_dir, new_nn_file, nngpt_dir, synth_dir
from ab.gpt.util.NNEval import NNEval
from ab.gpt.util.Util import verify_nn_code, copy_to_lemur

# --- Default Evaluation Parameters ---
# These will be used as defaults for argparse arguments
NN_TRAIN_EPOCHS = 1  # How many epochs to train the altered NN for evaluation
TASK = 'img-classification'
DATASET = 'cifar-10'
METRIC = 'acc'

# Default hyperparameters. 'epoch' will be overridden.
LR = 0.01
BATCH = 64
DROPOUT = 0.2
MOMENTUM = 0.9
TRANSFORM = 'norm_256_flip'  # A common default, used by NNEval if prm is None

SAVE_TO_DB = True
NN_NAME_PREFIX = None
NN_ALTER_EPOCHS = None
ONLY_EPOCH = None
EPOCH_LIMIT_MINUTES = None
CUSTOM_SYNTH_DIR = None


def main(nn_name_prefix=NN_NAME_PREFIX, nn_train_epochs=NN_TRAIN_EPOCHS, only_epoch=ONLY_EPOCH, save_to_db=SAVE_TO_DB,
         nn_alter_epochs=NN_ALTER_EPOCHS, task=TASK, dataset=DATASET, metric=METRIC, lr=LR, batch=BATCH, dropout=DROPOUT, momentum=MOMENTUM,
         transform=TRANSFORM, epoch_limit_minutes=EPOCH_LIMIT_MINUTES, custom_synth_dir=CUSTOM_SYNTH_DIR):
    base_nngpt_path = nngpt_dir  # out/nngpt
    if nn_alter_epochs is None:
        if epoch_dir().is_dir():
            nn_alter_epochs = len(os.listdir(epoch_dir()))
        else:
            print()
            print(f"Directory {epoch_dir()} doesn't exist", file=sys.stderr)

    if nn_alter_epochs:
        for i in ([only_epoch] if only_epoch is not None else range(nn_alter_epochs)):
            # Path to the output of one NNAlter.py epoch (e.g., out/nngpt/llm/epoch/A0)
            current_alter_epoch_path = epoch_dir(i)

            # If a custom custom_synth_dir is given, use it; otherwise build from epoch_dir
            if custom_synth_dir:
                models_base_dir = Path(custom_synth_dir)
            else:
                models_base_dir = synth_dir(current_alter_epoch_path)

            if not models_base_dir.exists():
                print(f"Directory {models_base_dir} for NNAlter epoch {i} not found. Skipping.")
                continue

            print(f"\n--- Scanning NNAlter Epoch Directory: {current_alter_epoch_path} ---")
            print(f"--- Synthesized Models Directory: {models_base_dir} ---")

            for model_folder_name in os.listdir(models_base_dir):
                model_dir_path = models_base_dir / model_folder_name
                if not model_dir_path.is_dir():
                    continue

                code_file_path = model_dir_path / new_nn_file
                df_file_path = model_dir_path / 'dataframe.df'  # Original model's metadata

                if not code_file_path.exists():
                    print(f"Code file {new_nn_file} not found in {model_dir_path}. Skipping.")
                    continue

                print(f"\n--- Evaluating Model: {model_dir_path.relative_to(base_nngpt_path)} ---")

                if not verify_nn_code(model_dir_path, code_file_path):
                    print(f"Code verification failed for {code_file_path}. Skipping evaluation.")
                    with open(model_dir_path / 'eval_verification_failed.txt', 'w') as f:
                        f.write("Initial code verification failed.")
                    continue

                # Initialize task, dataset, metric, and prm from command-line arguments (or their defaults)
                task = task
                dataset = dataset
                metric = metric
                # This prm structure is consistent with LEMUR dataset's expectations for model training
                prm = {
                    'lr': lr,
                    'batch': batch,
                    'dropout': dropout,
                    'momentum': momentum,
                    'transform': transform,  # Default transform from CLI
                    # 'epoch' will be set explicitly later
                }
                prefix_for_db = nn_name_prefix  # Default prefix
                origdf = None
                orig_pref = None
                if df_file_path.exists():
                    try:
                        origdf = pd.read_pickle(df_file_path)
                        # Override with values from dataframe.df if they exist
                        task = origdf.get('task', task)
                        dataset = origdf.get('dataset', dataset)
                        metric = origdf.get('metric', metric)
                        orig_pref = origdf['nn'].split('-')[0]

                        original_prm_from_df = origdf.get('prm')
                        if isinstance(original_prm_from_df, dict):
                            # Update prm with values from df, df values take precedence.
                            # This will update lr, batch, dropout, momentum, transform if they exist in original_prm_from_df,
                            # and also add any other model-specific hyperparameters from the original model.
                            prm.update(original_prm_from_df)

                        prefix_for_db = origdf.get('nn', prefix_for_db).split('-')[0]
                        print(f"  Loaded metadata from dataframe.df: task={task}, dataset={dataset}, metric={metric}")
                    except Exception as e:
                        print(f"  Error loading dataframe.df from {df_file_path}: {e}. Using command-line/default parameters for task, dataset, metric, and prm structure.")
                else:
                    print(f"  No dataframe.df found. Using command-line/default evaluation parameters.")

                # Crucial: set training epochs for this evaluation from nn_train_epochs
                # This overrides any 'epoch' value that might have come from original_prm_from_df
                prm['epoch'] = nn_train_epochs
                print(f"  Final parameters for NNEval: {prm}")
                print(f"  Task: {task}, Dataset: {dataset}, Metric: {metric}, Prefix: {prefix_for_db}")

                try:
                    evaluator = NNEval(
                        model_source_package=str(model_dir_path),
                        task=task,
                        dataset=dataset,
                        metric=metric,
                        prm=prm,  # Pass the constructed prm
                        save_to_db=save_to_db,
                        prefix=prefix_for_db,
                        save_path=model_dir_path
                    )
                    if epoch_limit_minutes:
                        evaluator.epoch_limit_minutes = epoch_limit_minutes
                    eval_results = evaluator.evaluate(code_file_path)
                    print(f"  Evaluation results for {model_folder_name}: {eval_results}")
                    eval_info_data = {
                        "eval_args": evaluator.get_args(),  # This will show the prm used by NNEval
                        "eval_results": eval_results,
                        "cli_args": {'task': task, 'dataset': dataset, "metric": metric, "lr": lr, "batch": batch,
                                     'dropout': dropout, 'momentum': momentum, 'transform': transform}
                    }
                    with open(model_dir_path / 'eval_info.json', 'w+') as f:
                        json.dump(eval_info_data, f, indent=4, default=str)

                    nn_name = uuid4(read_py_file_as_string(code_file_path))

                    pref = nn_name_prefix or orig_pref
                    if pref:
                        nn_name = pref + '-' + nn_name
                    copy_to_lemur(origdf, model_dir_path, nn_name)

                except Exception as e:
                    error_msg = f"Error evaluating model {model_folder_name}: {e}"
                    print(f"  {error_msg}")
                    detailed_error = traceback.format_exc()
                    print(detailed_error)
                    with open(model_dir_path / 'error.txt', 'w+') as f:
                        f.write(f"{error_msg}\n\n{detailed_error}")
                finally:
                    release_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neural Networks generated by NNAlter.py.")
    parser.add_argument('--nn_alter_epochs', type=int, default=NN_ALTER_EPOCHS,
                        help="Number of epochs NNAlter.py was run for (e.g., if NNAlter's -e was 8, use 8 here).")
    parser.add_argument('-nae', '--only_epoch', type=int, default=ONLY_EPOCH,
                        help="Run NNAlter.py for the specified epoch only.")
    parser.add_argument('-nte', '--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f"Number of epochs to train each altered NN during evaluation (default: {NN_TRAIN_EPOCHS}).")
    # Configurable evaluation parameters
    parser.add_argument('--task', type=str, default=TASK,
                        help=f"Default task for NNEval if not in dataframe.df (default: {TASK}).")
    parser.add_argument('--dataset', type=str, default=DATASET,
                        help=f"Default dataset for NNEval if not in dataframe.df (default: {DATASET}).")
    parser.add_argument('--metric', type=str, default=METRIC,
                        help=f"Default metric for NNEval if not in dataframe.df (default: {METRIC}).")

    # Configurable hyperparameters (part of prm dictionary for NNEval)
    parser.add_argument('--lr', type=float, default=LR,
                        help=f"Learning rate for NNEval if not in dataframe.df's prm (default: {LR}).")
    parser.add_argument('--batch_size', type=int, default=BATCH,
                        help=f"Batch size for NNEval if not in dataframe.df's prm (default: {BATCH}). Stored as 'batch' in prm.")
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help=f"Dropout rate for NNEval if not in dataframe.df's prm (default: {DROPOUT}).")
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help=f"Momentum for NNEval if not in dataframe.df's prm (default: {MOMENTUM}).")
    parser.add_argument('--transform', type=str, default=TRANSFORM,
                        help=f"Default transform for NNEval if not in dataframe.df's prm (default: {TRANSFORM}). Stored as 'transform' in prm.")

    # Other NNEval options
    parser.add_argument('--save_to_db', action=argparse.BooleanOptionalAction, default=SAVE_TO_DB,
                        help="Whether to save evaluation results to the database (enables with --save-to-db, disables with --no-save-to-db; default: enabled).")
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f"Default neural network name prefix (default: {NN_NAME_PREFIX}).")
    # Custom custom_synth_dir
    parser.add_argument('--custom_synth_dir', dest='custom_synth_dir', type=str, default=CUSTOM_SYNTH_DIR,
                        help="Custom directory containing generated models")
    parser.add_argument('--epoch_limit_minutes', type=int, default=EPOCH_LIMIT_MINUTES,
                        help="Max minutes allowed per epoch (default: specified in NN Dataset).")

    args = parser.parse_args()
    """
    Evaluates neural networks generated by NNAlter.py.

    :param args: Parsed command-line arguments.
    """
    print(f"Starting evaluation of altered NNs...")
    print(f"NNAlter run epochs to scan: {args.nn_alter_epochs}")
    print(f"Each altered NN will be trained for: {args.nn_train_epochs} epochs for evaluation.")
    print(f"Base task: {args.task}, Base dataset: {args.dataset}, Base metric: {args.metric}")
    print(f"Base Hyperparameters for NNEval (before df override):")
    print(f"  LR: {args.lr}, Batch Size: {args.batch_size}, Dropout: {args.dropout}, Momentum: {args.momentum}, Transform: {args.transform}")
    print(f"Save to DB: {args.save_to_db}")
    print(f"Prefix for the names of generated neural network: {args.nn_name_prefix}")

    main(args.nn_name_prefix, args.nn_train_epochs, args.transform.args.only_epoch, args.save_to_db, args.nn_alter_epochs,
         args.task, args.dataset, args.metric, args.lr, args.batch_size, args.dropout, args.momentum, args.epoch_limit_minutes,
         args.custom_synth_dir)
