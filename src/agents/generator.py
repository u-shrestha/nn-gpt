#!/usr/bin/env python3
"""
AI Agent System - Generator Node
Integrates nn-gpt LLM-based architecture generation with database caching.

This module provides the generator agent for a multi-agent AutoML system.
It generates neural network architectures using the university's fine-tuned LLM
and intelligently retrieves metrics from the database when models already exist.

Author: Hamza Naseem(AutoML Research Team)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import ab.nn as lemur
from ab.nn.util.Util import uuid4, read_py_file_as_string

from ab.gpt.util.Tune import nn_gen
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Const import conf_test_dir, conf_llm_dir, epoch_dir, synth_dir
from ab.gpt.util.Util import extract_code
from ab.nn.util.Util import create_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_metrics_from_database(
    model_code: str, 
    task: str, 
    dataset: str, 
    metric: str, 
    prm: dict,
    nn_train_epochs: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Query database for existing model metrics.
    
    Args:
        model_code: Python code of the neural network model
        task: Task type (e.g., 'img-classification')
        dataset: Dataset name (e.g., 'cifar-10')
        metric: Metric name (e.g., 'acc')
        prm: Hyperparameters dictionary
        nn_train_epochs: Number of training epochs to look for
        
    Returns:
        Dictionary with model metrics if found, None otherwise
    """
    try:
        checksum = uuid4(model_code)
        lemur.data.cache_clear()
        df = lemur.data(only_best_accuracy=False)
        
        # Find matching models
        matches = df[
            (df['nn_id'] == checksum) &
            (df['task'] == task) &
            (df['dataset'] == dataset) &
            (df['metric'] == metric)
        ]
        
        if matches.empty:
            logger.info(f"No matching model found in database for checksum {checksum[:8]}...")
            return None
        
        # Try to find exact epoch match
        epoch_data = matches[matches['epoch'] == nn_train_epochs]
        if epoch_data.empty:
            # Fallback: use closest epoch <= requested
            available_epochs = matches['epoch'].unique()
            valid_epochs = [e for e in available_epochs if e <= nn_train_epochs]
            use_epoch = max(valid_epochs) if valid_epochs else min(available_epochs)
            epoch_data = matches[matches['epoch'] == use_epoch]
            logger.info(f"Exact epoch {nn_train_epochs} not found, using epoch {use_epoch}")
        
        if not epoch_data.empty:
            row = epoch_data.iloc[0]
            accuracy = float(row['accuracy'])
            accuracy_to_time = 0.0
            
            # Calculate accuracy_to_time if duration available
            if 'duration' in row and pd.notna(row['duration']):
                duration_sec = float(row['duration']) / 1e9  # Convert nanoseconds to seconds
                if duration_sec > 0:
                    accuracy_to_time = accuracy / duration_sec
            
            metrics = {
                "model_name": row.get('nn', checksum),
                "accuracy": accuracy,
                "accuracy_to_time": accuracy_to_time,
                "score": accuracy,
                "epoch": int(row['epoch']),
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "prm": row.get('prm', prm),
            }
            logger.info(f"Found metrics in database: accuracy={accuracy:.4f}, epoch={metrics['epoch']}")
            return metrics
            
        return None
        
    except Exception as e:
        logger.error(f"Error querying database: {e}", exc_info=True)
        return None


def create_eval_info_from_database_metrics(
    metrics: Dict[str, Any], 
    model_dir: Path
) -> Dict[str, Any]:
    """
    Create eval_info.json file from database metrics.
    
    Args:
        metrics: Dictionary containing model metrics from database
        model_dir: Directory where eval_info.json should be saved
        
    Returns:
        Dictionary with eval_info structure
    """
    eval_info = {
        "eval_args": {
            "task": metrics['task'],
            "dataset": metrics['dataset'],
            "metric": metrics['metric'],
            "prm": metrics['prm'],
        },
        "eval_results": [
            metrics['model_name'],
            metrics['accuracy'],
            metrics['accuracy_to_time'],
            metrics['score'],
        ],
        "cli_args": {
            "task": metrics['task'],
            "dataset": metrics['dataset'],
            "metric": metrics['metric'],
        },
        "source": "database",
        "epoch": metrics['epoch'],
    }
    
    eval_file = model_dir / "eval_info.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_info, f, indent=4, default=str)
    
    logger.info(f"Created eval_info.json from database metrics at {eval_file}")
    return eval_info


def generator_node_with_db_metrics(
    state: Dict[str, Any],
    llm_conf: str = 'nngpt_ds_coder_1.3b_instruct.json',
    nn_gen_conf: str = 'NN_gen.json',
    nn_gen_conf_id: str = 'improve_classification_only',
    max_new_tokens: int = 12 * 1024,  # 12,288 tokens - maximum for 1.3B model
    nn_train_epochs: int = 2,
    test_nn: int = 1,
    temperature: float = 0.8,  # Professor's default for generation diversity
    top_k: int = 70,          # Professor's default for sampling
    top_p: float = 0.9,       # Professor's default for nucleus sampling
) -> Dict[str, Any]:
    """
    Generator agent node with database fallback for AutoML multi-agent system.
    
    This function:
    1. Loads and initializes the LLM (DeepSeek Coder 1.3B)
    2. Generates neural network architectures using nn_gen()
    3. Handles extraction failures by parsing full_output.txt
    4. Queries database for metrics if model already exists (avoids redundant training)
    5. Returns structured state with model code and metrics
    
    Args:
        state: Agent state dictionary containing:
            - experiment_id (str): Unique identifier for this experiment
            - spec (str, optional): Task specification
            - dataset (str, optional): Target dataset name
            - task (str, optional): Task type (default: 'img-classification')
            - metric (str, optional): Evaluation metric (default: 'acc')
        llm_conf: LLM configuration file name
        nn_gen_conf: Neural network generation config file name
        nn_gen_conf_id: Configuration key to use from nn_gen_conf
        max_new_tokens: Maximum tokens for LLM generation (12K for 1.3B model)
        nn_train_epochs: Number of epochs for quick training
        test_nn: Number of models to generate
        temperature: LLM temperature for sampling diversity
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        
    Returns:
        Updated state dictionary with:
            - status (str): 'success' or 'failed'
            - model_code (str): Generated PyTorch model code
            - hyperparameters (dict): Extracted hyperparameters
            - early_metrics (dict): Training metrics (accuracy, loss, etc.)
            - metrics_source (str): 'eval_info.json', 'database', or 'none'
            - error_message (str, optional): Error details if failed
            
    Example:
        >>> state = {
        ...     "experiment_id": "exp_001",
        ...     "spec": "Image classification on CIFAR-10",
        ...     "dataset": "cifar-10"
        ... }
        >>> result = generator_node_with_db_metrics(state)
        >>> print(result['status'])  # 'success'
        >>> print(result['early_metrics']['accuracy_2epoch'])  # 0.85
    """
    
    # Validate required state keys
    required_keys = ['experiment_id']
    missing_keys = [k for k in required_keys if k not in state]
    if missing_keys:
        error_msg = f"Missing required state keys: {missing_keys}"
        logger.error(error_msg)
        return {
            **state,
            "status": "failed",
            "error_message": error_msg
        }
    
    try:
        epoch = 0
        
        # Extract configuration from state with defaults
        task = state.get('task', 'img-classification')
        dataset = state.get('dataset', 'cifar-10')
        metric = state.get('metric', 'acc')
        
        logger.info(f"Starting generator for experiment: {state['experiment_id']}")
        logger.info(f"Configuration: task={task}, dataset={dataset}, metric={metric}")
        
        # Load LLM configuration
        llm_config_path = conf_llm_dir / llm_conf
        with open(llm_config_path) as f:
            llm_config = json.load(f)
        
        base_model_name = llm_config['base_model_name']
        access_token = None
        use_deepspeed = llm_config.get('use_deepspeed', False)
        context_length = llm_config.get('context_length')
        
        logger.info(f"Using model: {base_model_name}")
        
        # Load prompt configuration
        prompt_config_path = conf_test_dir / nn_gen_conf
        with open(prompt_config_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        
        # Initialize LLM
        logger.info("Loading LLM and tokenizer...")
        model_loader = LLM(
            base_model_name,
            quantization_config_4bit,
            access_token=access_token,
            use_deepspeed=use_deepspeed,
            context_length=context_length
        )
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        # Initialize ChatBot with professor's defaults
        chat_bot = ChatBot(
            model, tokenizer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        logger.info("LLM initialized successfully")
        
        # Prepare generation parameters
        out_path = epoch_dir(epoch)
        conf_keys = (nn_gen_conf_id,)
        
        # Generate model using nn_gen
        logger.info(f"Generating model with config: {nn_gen_conf_id}")
        nn_gen(
            epoch=epoch,
            out_path=out_path,
            chat_bot=chat_bot,
            conf_keys=conf_keys,
            nn_train_epochs=nn_train_epochs,
            prompt_dict=prompt_dict,
            test_nn=test_nn,
            max_new_tokens=max_new_tokens,
            save_llm_output=True,
            nn_name_prefix=state.get('experiment_id', None)
        )
        
        # ============================================================
        # FALLBACK: Extract model code from full_output.txt if missing
        # ============================================================
        models_dir = synth_dir(out_path)
        model_dir = models_dir / "B0"
        model_file = model_dir / "new_nn.py"
        eval_file = model_dir / "eval_info.json"
        full_output_file = model_dir / "full_output.txt"
        
        # Attempt to extract code if model file doesn't exist
        if (not model_file.exists() or model_file.stat().st_size == 0) and full_output_file.exists():

            logger.warning("Model file not found, attempting extraction from full_output.txt")
            
            with open(full_output_file, 'r', encoding='utf-8') as f:
                full_output = f.read()
            
            model_code = extract_code(full_output)
            
            if model_code and model_code.strip():
                logger.info("Successfully extracted model code from full_output.txt")
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(model_file, 'w', encoding='utf-8') as f:
                    f.write(model_code)
                logger.info(f"Saved extracted model code to {model_file}")
            else:
                logger.error("Could not extract model code from full_output.txt")
                return {
                    **state,
                    "status": "failed",
                    "error_message": "Model code extraction failed. LLM may have generated invalid output."
                }
        
        # Verify model file exists
        if not model_file.exists():
            error_msg = f"Model file not found: {model_file}"
            logger.error(error_msg)
            return {
                **state,
                "status": "failed",
                "error_message": error_msg
            }
        
        # Read generated model code
        with open(model_file, 'r', encoding='utf-8') as f:
            model_code = f.read()
        logger.info(f"Model code loaded: {len(model_code)} characters")
        
        # Extract hyperparameters
        hp_file = model_dir / "hp.txt"
        hyperparameters = {}
        
        if hp_file.exists():
            with open(hp_file, 'r', encoding='utf-8') as f:
                hyperparameters = json.load(f)
            logger.info(f"Hyperparameters loaded: {hyperparameters}")
        else:
            # Fallback: try to extract from full_output.txt
            if full_output_file.exists():
                from ab.gpt.util.Util import extract_hyperparam
                with open(full_output_file, 'r', encoding='utf-8') as f:
                    full_output = f.read()
                hp_text = extract_hyperparam(full_output)
                
                if hp_text and hp_text.strip():
                    try:
                        hyperparameters = json.loads(hp_text.replace("'", '"'))
                        # Save extracted hyperparameters
                        with open(hp_file, 'w', encoding='utf-8') as f:
                            json.dump(hyperparameters, f)
                        logger.info("Extracted and saved hyperparameters from full_output.txt")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Could not parse hyperparameters: {e}")
        
        # Read metadata from dataframe.df if available
        df_file = model_dir / "dataframe.df"
        
        if df_file.exists():
            import pickle
            try:
                with open(df_file, 'rb') as f:
                    origdf = pickle.load(f)
                    task = origdf.get('task', task)
                    dataset = origdf.get('dataset', dataset)
                    metric = origdf.get('metric', metric)
                logger.info(f"Metadata loaded from dataframe: task={task}, dataset={dataset}")
            except Exception as e:
                logger.warning(f"Could not load metadata from dataframe.df: {e}")
        
        # Check for evaluation metrics
        early_metrics = {}
        eval_data = {}
        metrics_source = "none"
        
        if eval_file.exists():
            # Metrics from evaluation
            logger.info("Found eval_info.json from training")
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            if eval_data.get("eval_results"):
                results = eval_data["eval_results"]
                if isinstance(results, list) and len(results) >= 4:
                    early_metrics = {
                        "accuracy_2epoch": results[1],
                        "accuracy_to_time": results[2],
                        "score": results[3],
                    }
                    metrics_source = "eval_info.json"
                    logger.info(f"Metrics from training: accuracy={early_metrics['accuracy_2epoch']:.4f}")
        else:
            # Fallback: query database for cached metrics
            logger.info("eval_info.json not found, querying database for cached metrics")
            
            db_metrics = get_metrics_from_database(
                model_code=model_code,
                task=task,
                dataset=dataset,
                metric=metric,
                prm=hyperparameters,
                nn_train_epochs=nn_train_epochs
            )
            
            if db_metrics:
                logger.info("Found metrics in database (model already exists)")
                eval_data = create_eval_info_from_database_metrics(db_metrics, model_dir)
                early_metrics = {
                    "accuracy_2epoch": db_metrics['accuracy'],
                    "accuracy_to_time": db_metrics['accuracy_to_time'],
                    "score": db_metrics['score'],
                }
                metrics_source = "database"
            else:
                logger.warning("No metrics found - model is new and training may have failed")
                metrics_source = "none"
        
        # Return success state
        result = {
            **state,
            "status": "success",
            "model_file_path": str(model_file),
            "model_code": model_code,
            "hyperparameters": hyperparameters,
            "eval_file_path": str(eval_file),
            "raw_eval_info": eval_data,
            "early_metrics": early_metrics,
            "epoch_used": epoch,
            "accuracy_2epoch": early_metrics.get("accuracy_2epoch"),
            "has_training_metrics": bool(early_metrics),
            "metrics_source": metrics_source,
        }
        
        logger.info(f"Generator completed successfully: metrics_source={metrics_source}")
        return result
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Generator failed with exception: {e}")
        logger.error(error_trace)
        
        return {
            **state,
            "status": "failed",
            "error_message": str(e),
            "traceback": error_trace
        }


if __name__ == "__main__":
    # Test the generator with sample state
    state = {
        "experiment_id": "test_gen",
        "spec": "Image classification",
        "dataset": "cifar-10",
        "task": "img-classification",
        "metric": "acc"
    }
    
    logger.info("Running generator test...")
    result = generator_node_with_db_metrics(state)
    
    print("\n" + "="*60)
    print("GENERATION RESULT:")
    print("="*60)
    print(json.dumps({
        "status": result.get("status"),
        "metrics_source": result.get("metrics_source"),
        "has_metrics": result.get("has_training_metrics"),
        "accuracy": result.get("accuracy_2epoch"),
    }, indent=2))
    
    if result.get("status") == "failed":
        print("\nERROR:")
        print(result.get("error_message"))