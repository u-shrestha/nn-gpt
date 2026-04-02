"""
Validates LLM outputs for the dataset-comparison classification task.
No NN training — just checks if the predicted dataset name matches ground truth.
"""
import json, os
from pathlib import Path
import pandas as pd

from ab.gpt.util.Const import new_out_file

import re

def extract_answer(llm_output: str, valid_datasets: list) -> str | None:
    """Find the first valid dataset name mentioned in the LLM output."""
    if not llm_output:
        return None
    text = llm_output.strip().lower()
    # Exact match first (handles clean one-word responses)
    for ds in valid_datasets:
        ds_lower = ds.lower() #new

        if text == ds.lower():
            return ds
        if ds_lower in text:
            return ds

        ds_normalized = ds_lower.replace('-', ' ').replace('_', ' ')
        if ds_normalized in text:
            return ds
    return None
    # Substring fallback (handles "Answer: cifar-100" etc.)
    for ds in valid_datasets:
        if re.search(r'\b' + re.escape(ds.lower()) + r'\b', text):
            return ds
        # if ds.lower() in text:
        #     return ds
    return None


def evaluate_epoch(models_base_dir: Path) -> dict:
    """
    Scan all B* directories, compare LLM answer to ground truth better_dataset.
    Writes classification_eval.json per model and returns aggregate stats.
    """
    results = []

    for model_id in sorted(os.listdir(models_base_dir)):
        model_dir = models_base_dir / model_id
        if not model_dir.is_dir():
            continue

        out_file = model_dir / new_out_file
        df_file  = model_dir / 'dataframe.df'

        if not out_file.exists():
            print(f'  [ClassificationEval] No output file for {model_id}, skipping.')
            continue
        if not df_file.exists():
            print(f'  [ClassificationEval] No dataframe.df for {model_id}, skipping.')
            continue

        try:
            raw_output   = out_file.read_text()
            # Strip prompt leakage: keep only text after the response marker
            for marker in ('### Response:', '<|im_start|>assistant', '<|assistant|>', '[/INST]'):
                if marker in raw_output:
                    raw_output = raw_output.split(marker, 1)[-1]
                    break
            llm_output   = raw_output.strip()
            origdf       = pd.read_pickle(df_file)
            ground_truth = origdf.get('better_dataset')
            dataset_1    = origdf.get('dataset')
            dataset_2    = origdf.get('dataset_2')

            valid     = [d for d in [dataset_1, dataset_2] if d]
            predicted = extract_answer(llm_output, valid)
            correct   = bool(predicted and ground_truth and predicted == ground_truth)

            per_model = {
                'model_id':     model_id,
                'predicted':    predicted,
                'ground_truth': ground_truth,
                'correct':      correct,
                'dataset_1':    dataset_1,
                'dataset_2':    dataset_2,
                'raw_output':   llm_output[:200] if llm_output else '',
            }
            results.append(per_model)

            with open(model_dir / 'classification_eval.json', 'w') as f:
                json.dump(per_model, f, indent=4)

            status = '✓' if correct else '✗'
            print(f'  [{status}] {model_id}: predicted={predicted!r}  truth={ground_truth!r}')

        except Exception as e:
            print(f'  [ClassificationEval] Error for {model_id}: {e}')

    total    = len(results)
    n_correct = sum(1 for r in results if r['correct'])
    accuracy = n_correct / total if total > 0 else 0.0

    print(f'\n  [ClassificationEval] Accuracy: {n_correct}/{total} = {accuracy:.3f}')
    return {'accuracy': accuracy, 'correct': n_correct, 'total': total, 'results': results}
