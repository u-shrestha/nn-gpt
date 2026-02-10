"""
Learning Rate Scheduler Utility Module

This module provides:
- unique_nn / unique_nn_cls functions for filtering models
- Database functions for storing/retrieving scheduler metrics
- Class-specific data generation and retrieval
- Hyperparameter extraction and validation
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import pandas as pd

from .const import (
    ACTIVE_MODEL,
    IMAGE_CLASSIFICATION_MODELS,
    ADDITIONAL_MODELS,
    ALL_MODELS,
    CIFAR10_CLASSES,
    CIFAR10_CLASSES_COUNT,
    CIFAR10_EXTENDED_CLASSES,
    DB_CONFIG,
    DEFAULT_HYPERPARAMS,
    OUTPUT_DIRS,
    PROMPT_CONFIG,
)

# ============================================================================
# 1. Neural Network Unique Functions
# ============================================================================

def unique_nn(
    epoch_max: int,
    nns: List[str],
    dataset: str = 'cifar-10',
    task: str = 'img-classification',
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Retrieve unique neural network models from database.
    
    Filters models by:
    - Epoch count
    - Model names (nns parameter)
    - Dataset
    - Task type
    - Metric for evaluation
    
    Args:
        epoch_max: Maximum number of epochs to filter by
        nns: List of neural network names to include
        dataset: Dataset name (default: 'cifar-10')
        task: Task type (default: 'img-classification')
        metric: Evaluation metric (default: 'accuracy')
    
    Returns:
        pd.DataFrame: Sorted by metric (descending), containing:
            - model_name
            - scheduler_type
            - metric_value
            - hyperparameters
            - epoch
    """
    
    # Query database for models
    query = f"""
        SELECT 
            model_name,
            scheduler_type,
            {metric} as metric_value,
            hyperparameters,
            epoch
        FROM scheduler_results
        WHERE 
            epoch <= ?
            AND model_name IN ({','.join(['?' for _ in nns])})
            AND dataset = ?
            AND task = ?
            AND best_accuracy = 1
        ORDER BY {metric} DESC
    """
    
    params = [epoch_max] + nns + [dataset, task]
    
    try:
        conn = _get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            print(f"⚠️ No results found for models: {nns}")
            return pd.DataFrame()
        
        print(f"✅ Retrieved {len(df)} unique models from database")
        return df.sort_values(by='metric_value', ascending=False)
    
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return pd.DataFrame()


def unique_nn_cls(
    epoch_max: int,
    dataset: str = 'cifar-10',
    task: str = 'img-classification',
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Retrieve unique neural network classification models.
    
    Specialized wrapper for unique_nn() that filters to only
    image classification models from the core model set.
    
    Args:
        epoch_max: Maximum epochs to filter by
        dataset: Dataset name (default: 'cifar-10')
        task: Task type (default: 'img-classification')
        metric: Evaluation metric (default: 'accuracy')
    
    Returns:
        pd.DataFrame: Core classification models sorted by metric
    """
    
    return unique_nn(
        epoch_max=epoch_max,
        nns=IMAGE_CLASSIFICATION_MODELS,
        dataset=dataset,
        task=task,
        metric=metric
    )


def get_active_model() -> str:
    """
    Get the currently active model for fine-tuning.
    
    Returns:
        str: Active model name (e.g., 'ResNet')
    """
    return ACTIVE_MODEL


def set_active_model(model_name: str) -> bool:
    """
    Set the active model for fine-tuning.
    
    Args:
        model_name: Name of the model to activate
    
    Returns:
        bool: True if successful, False if model not found
    """
    if model_name not in ALL_MODELS:
        print(f"❌ Model '{model_name}' not found in supported models")
        print(f"   Supported: {ALL_MODELS}")
        return False
    
    globals()['ACTIVE_MODEL'] = model_name
    print(f"✅ Active model set to: {model_name}")
    return True


# ============================================================================
# 2. Database Functions
# ============================================================================

def _get_db_connection():
    """
    Get SQLite database connection.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    db_path = DB_CONFIG['path']
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def init_database() -> bool:
    """
    Initialize the scheduler results database.
    Creates tables if they don't exist.
    
    Returns:
        bool: True if successful
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        # Create scheduler_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                scheduler_type TEXT NOT NULL,
                dataset TEXT NOT NULL,
                task TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                accuracy REAL,
                loss REAL,
                best_accuracy INTEGER DEFAULT 0,
                hyperparameters TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create class_data table for class-specific metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS class_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                scheduler_type TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                epoch INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON scheduler_results(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scheduler_type ON scheduler_results(scheduler_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON scheduler_results(dataset)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_data ON class_data(model_name, class_id)")
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False


def save_scheduler_result(
    model_name: str,
    scheduler_type: str,
    dataset: str,
    task: str,
    epoch: int,
    accuracy: float,
    loss: float,
    best_accuracy: bool = False,
    hyperparameters: Optional[Dict] = None
) -> bool:
    """
    Save scheduler training results to database.
    
    Args:
        model_name: Name of the model
        scheduler_type: Type of LR scheduler used
        dataset: Dataset name
        task: Task type
        epoch: Epoch number
        accuracy: Accuracy metric
        loss: Loss value
        best_accuracy: Whether this is the best accuracy so far
        hyperparameters: Dictionary of hyperparameters used
    
    Returns:
        bool: True if successful
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        hp_json = json.dumps(hyperparameters) if hyperparameters else "{}"
        
        cursor.execute("""
            INSERT INTO scheduler_results
            (model_name, scheduler_type, dataset, task, epoch, accuracy, loss, best_accuracy, hyperparameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            scheduler_type,
            dataset,
            task,
            epoch,
            accuracy,
            loss,
            int(best_accuracy),
            hp_json
        ))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving scheduler result: {e}")
        return False


def get_best_schedulers(
    model_name: str,
    dataset: str = 'cifar-10',
    task: str = 'img-classification',
    limit: int = 5
) -> pd.DataFrame:
    """
    Get the best performing schedulers for a model.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name
        task: Task type
        limit: Maximum number of results to return
    
    Returns:
        pd.DataFrame: Best schedulers sorted by accuracy
    """
    query = """
        SELECT 
            scheduler_type,
            accuracy,
            loss,
            epoch,
            hyperparameters
        FROM scheduler_results
        WHERE 
            model_name = ?
            AND dataset = ?
            AND task = ?
            AND best_accuracy = 1
        ORDER BY accuracy DESC
        LIMIT ?
    """
    
    try:
        conn = _get_db_connection()
        df = pd.read_sql_query(query, conn, params=(model_name, dataset, task, limit))
        conn.close()
        return df
    except Exception as e:
        print(f"❌ Error retrieving best schedulers: {e}")
        return pd.DataFrame()


# ============================================================================
# 3. Class-Specific Data Functions
# ============================================================================

def init_class_data(
    model_name: str,
    scheduler_type: str,
    num_classes: int = 10,
    epoch: int = 0
) -> bool:
    """
    Initialize class-specific data for a model-scheduler combination.
    
    Generates entries for all CIFAR-10 classes (or extended classes if enabled).
    
    Args:
        model_name: Name of the model
        scheduler_type: Type of LR scheduler
        num_classes: Number of classes (10 for CIFAR-10, 20 for extended)
        epoch: Starting epoch
    
    Returns:
        bool: True if successful
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        # Use appropriate class list
        classes = CIFAR10_CLASSES if num_classes == 10 else CIFAR10_EXTENDED_CLASSES
        
        for class_id, class_name in enumerate(classes[:num_classes]):
            cursor.execute("""
                INSERT OR IGNORE INTO class_data
                (model_name, scheduler_type, class_id, class_name, accuracy, epoch)
                VALUES (?, ?, ?, ?, 0.0, ?)
            """, (model_name, scheduler_type, class_id, class_name, epoch))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Initialized class data for {model_name} with {scheduler_type} ({num_classes} classes)")
        return True
    
    except Exception as e:
        print(f"❌ Error initializing class data: {e}")
        return False


def save_class_accuracy(
    model_name: str,
    scheduler_type: str,
    class_id: int,
    class_name: str,
    accuracy: float,
    precision: float = 0.0,
    recall: float = 0.0,
    f1_score: float = 0.0,
    epoch: int = 0
) -> bool:
    """
    Save per-class accuracy metrics.
    
    Args:
        model_name: Name of the model
        scheduler_type: Type of LR scheduler
        class_id: Class ID (0-9 for CIFAR-10)
        class_name: Class name
        accuracy: Per-class accuracy
        precision: Per-class precision
        recall: Per-class recall
        f1_score: Per-class F1 score
        epoch: Epoch number
    
    Returns:
        bool: True if successful
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO class_data
            (model_name, scheduler_type, class_id, class_name, accuracy, precision, recall, f1_score, epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            scheduler_type,
            class_id,
            class_name,
            accuracy,
            precision,
            recall,
            f1_score,
            epoch
        ))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving class accuracy: {e}")
        return False


def get_class_accuracies(
    model_name: str,
    scheduler_type: str,
    epoch: int = 0
) -> pd.DataFrame:
    """
    Retrieve per-class accuracy metrics.
    
    Args:
        model_name: Name of the model
        scheduler_type: Type of LR scheduler
        epoch: Epoch number to retrieve
    
    Returns:
        pd.DataFrame: Class-specific metrics
    """
    query = """
        SELECT 
            class_id,
            class_name,
            accuracy,
            precision,
            recall,
            f1_score
        FROM class_data
        WHERE 
            model_name = ?
            AND scheduler_type = ?
            AND epoch = ?
        ORDER BY class_id ASC
    """
    
    try:
        conn = _get_db_connection()
        df = pd.read_sql_query(query, conn, params=(model_name, scheduler_type, epoch))
        conn.close()
        return df
    except Exception as e:
        print(f"❌ Error retrieving class accuracies: {e}")
        return pd.DataFrame()


def generate_class_data_for_epoch(
    model_name: str,
    scheduler_type: str,
    epoch: int,
    num_classes: int = 10
) -> Dict[str, float]:
    """
    Generate synthetic class-specific data for an epoch.
    
    Currently generates 10 classes. Should be extended for 20 classes.
    
    Args:
        model_name: Name of the model
        scheduler_type: Type of LR scheduler
        epoch: Epoch number
        num_classes: Number of classes to generate (10 or 20)
    
    Returns:
        Dict: Class accuracies indexed by class name
    """
    
    classes = CIFAR10_CLASSES if num_classes == 10 else CIFAR10_EXTENDED_CLASSES
    class_data = {}
    
    try:
        conn = _get_db_connection()
        
        for class_id, class_name in enumerate(classes[:num_classes]):
            # Generate synthetic but reasonable accuracy
            # In production, this would come from actual training metrics
            base_accuracy = 0.7 + (epoch * 0.01)  # Improves with epoch
            class_accuracy = base_accuracy + (hash(f"{model_name}_{class_name}") % 20) / 100
            class_accuracy = min(max(class_accuracy, 0.0), 1.0)  # Clamp to [0, 1]
            
            save_class_accuracy(
                model_name=model_name,
                scheduler_type=scheduler_type,
                class_id=class_id,
                class_name=class_name,
                accuracy=class_accuracy,
                epoch=epoch
            )
            
            class_data[class_name] = class_accuracy
        
        print(f"✅ Generated class data for {model_name}/{scheduler_type} epoch {epoch}")
        return class_data
    
    except Exception as e:
        print(f"❌ Error generating class data: {e}")
        return {}


# ============================================================================
# 4. Hyperparameter Functions
# ============================================================================

def get_hyperparams(scheduler_name: str) -> Dict:
    """
    Get default hyperparameters for a scheduler.
    
    Args:
        scheduler_name: Name of the scheduler
    
    Returns:
        Dict: Hyperparameters for the scheduler
    """
    hp = DEFAULT_HYPERPARAMS.copy()
    
    # Customize based on scheduler type
    if 'OneCycle' in scheduler_name:
        hp['pct_start'] = 0.3
        hp['div_factor'] = 25.0
    elif 'CosineAnnealing' in scheduler_name:
        hp['eta_min'] = 0.0
    elif 'MultiStep' in scheduler_name:
        hp['milestone0'] = 0.3
        hp['milestone1'] = 0.6
        hp['milestone2'] = 0.8
    
    return hp


def validate_hyperparams(hp: Dict) -> Tuple[bool, str]:
    """
    Validate hyperparameters.
    
    Args:
        hp: Hyperparameters dictionary
    
    Returns:
        Tuple: (is_valid, error_message)
    """
    required = ['learning_rate', 'momentum', 'dropout', 'epoch_max']
    
    for param in required:
        if param not in hp:
            return False, f"Missing required parameter: {param}"
    
    # Validate ranges
    if not (0 < hp['learning_rate'] <= 1.0):
        return False, f"learning_rate must be in (0, 1.0], got {hp['learning_rate']}"
    
    if not (0 <= hp['dropout'] < 1.0):
        return False, f"dropout must be in [0, 1.0), got {hp['dropout']}"
    
    if hp['epoch_max'] <= 0:
        return False, f"epoch_max must be positive, got {hp['epoch_max']}"
    
    return True, "Valid"


# ============================================================================
# 5. Summary Functions
# ============================================================================

def print_model_summary():
    """Print summary of supported models and schedulers."""
    print("\n" + "="*70)
    print("SUPPORTED MODELS & SCHEDULERS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Active Model: {get_active_model()}")
    print(f"📊 Total Classification Models: {len(IMAGE_CLASSIFICATION_MODELS)}")
    print(f"📊 Additional Models Available: {len(ADDITIONAL_MODELS)}")
    print(f"📊 CIFAR-10 Classes: {CIFAR10_CLASSES_COUNT}")
    
    print(f"\n🎯 Classification Models:")
    for model in IMAGE_CLASSIFICATION_MODELS:
        print(f"   - {model}")
    
    print(f"\n🔮 Additional Models (Future Support):")
    for model in ADDITIONAL_MODELS:
        print(f"   - {model}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test initialization
    init_database()
    print_model_summary()
    print(f"✅ Utility module ready. Active model: {get_active_model()}")
