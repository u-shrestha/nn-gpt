# Learning Rate Scheduler Configuration Module - Implementation Guide

## Overview

This implementation provides a complete workflow for:
1. **Script & Model Configuration** - Define and manage model selection with prefix-based logic
2. **File & Class Structure** - Comprehensive model management using `const.py`
3. **Data Generation & Database** - Class-specific data and database integration via `util.py`

## Files Created

### 1. `const.py` - Configuration Constants
Location: `/home/hafsamateen/Project_ResNet/nn-gpt/ab/gpt/brute/lr/const.py`

**Key Components:**
```python
# Model Configuration
ACTIVE_MODEL = "ResNet"
IMAGE_CLASSIFICATION_MODELS = [...]  # 25 core models
ADDITIONAL_MODELS = [...]             # 8 additional models (future support)
ALL_MODELS = IMAGE_CLASSIFICATION_MODELS + ADDITIONAL_MODELS

# Scheduler Types
SCHEDULER_TYPES = {
    'step_based': [...],
    'polynomial': [...],
    'cosine': [...],
    'cyclic': [...],
    'warmup': [...],
    'lambda': [...],
    'multiplicative': [...],
}

# Database Configuration
DB_CONFIG = {
    'path': 'db/ab.nn.db',
    'use_cache': True,
    'auto_refresh': True,
}

# CIFAR-10 Classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
```

**Features:**
- [x] Image classification models list (25 models)
- [x] Support for ResNet as primary focus
- [x] Extension points for additional models
- [x] Database path configuration
- [x] CIFAR-10 class definitions (10 classes)
- [x] Feature flags for future extensions

### 2. `util.py` - Utility Functions
Location: `/home/hafsamateen/Project_ResNet/nn-gpt/ab/gpt/brute/lr/util.py`

**Key Functions:**

#### Neural Network Filtering
```python
unique_nn(epoch_max, nns, dataset, task, metric) -> pd.DataFrame
    """Retrieve unique NN models from database"""

unique_nn_cls(epoch_max, dataset, task, metric) -> pd.DataFrame
    """Retrieve unique classification models (ResNet, VGG, etc.)"""
```

#### Model Management
```python
get_active_model() -> str
    """Get currently active model for fine-tuning"""

set_active_model(model_name: str) -> bool
    """Set active model for fine-tuning"""
```

#### Database Functions
```python
init_database() -> bool
    """Initialize scheduler_results and class_data tables"""

save_scheduler_result(model_name, scheduler_type, dataset, task, epoch, 
                     accuracy, loss, best_accuracy, hyperparameters) -> bool
    """Save training results to database"""

get_best_schedulers(model_name, dataset, task, limit) -> pd.DataFrame
    """Get best performing schedulers for a model"""
```

#### Class-Specific Data
```python
init_class_data(model_name, scheduler_type, num_classes, epoch) -> bool
    """Initialize per-class data storage"""

save_class_accuracy(model_name, scheduler_type, class_id, class_name,
                   accuracy, precision, recall, f1_score, epoch) -> bool
    """Save per-class metrics"""

get_class_accuracies(model_name, scheduler_type, epoch) -> pd.DataFrame
    """Retrieve per-class metrics"""

generate_class_data_for_epoch(model_name, scheduler_type, epoch, 
                             num_classes) -> Dict[str, float]
    """Generate class-specific data for an epoch"""
```

#### Hyperparameter Management
```python
get_hyperparams(scheduler_name: str) -> Dict
    """Get default hyperparameters for scheduler"""

validate_hyperparams(hp: Dict) -> Tuple[bool, str]
    """Validate hyperparameter ranges"""
```

### 3. `schedulers.py` - Enhanced Main Script
Location: `/home/hafsamateen/Project_ResNet/nn-gpt/ab/gpt/brute/lr/schedulers.py`

**New Sections Added:**
1. **Import Configuration** - Imports const and util modules
2. **Model Setup Phase** - Initializes database and prints model summary
3. **Fine-Tuning Configuration** - Initializes class data for active model
4. **Database Logging** - Saves scheduler results during generation
5. **Enhanced Summary** - Shows active model, dataset, and next steps

**Integration Points:**
```python
# Configuration loading
if CONFIG_AVAILABLE:
    init_database()
    print_model_summary()
    OUTPUT_DIR = str(OUTPUT_DIRS['base'])
    BASE_MODEL_NAME = FILE_CONVENTIONS['model_prefix']

# Fine-tuning setup
if CONFIG_AVAILABLE:
    active_model = get_active_model()
    for scheduler_idx in range(1, 35):
        init_class_data(...)

# Database logging during generation
if CONFIG_AVAILABLE:
    hp_dict = get_hyperparams(strategy['name'])
    save_scheduler_result(...)
```

### 4. `__init__.py` - Package Initialization
Location: `/home/hafsamateen/Project_ResNet/nn-gpt/ab/gpt/brute/lr/__init__.py`

**Exports:**
- Configuration constants
- Utility functions
- Database operations
- Class data management functions

## Implementation Details

### Section 1: Script & Model Configuration

**Prefix Model Logic:**
- Models use `A0_` prefix for directory naming
- Active model determined by `ACTIVE_MODEL` in const.py
- Currently focused on ResNet

**Fine-Tuning Workflow:**
```python
# Prompts & Model Detection
PROMPT_CONFIG = {
    'model_detection': {
        'prefix': 'model_',
        'active_indicator': '<<<ACTIVE>>>',
        'keywords': ['ResNet', 'checkpoint', 'fine-tune']
    },
    'lr_extraction': {
        'keywords': ['learning_rate', 'lr'],
        'default_lr': 0.1,
        'lr_range': [0.001, 1.0]
    }
}
```

**Action Item:** Support for additional models beyond ResNet
- Models defined in `ADDITIONAL_MODELS`
- Enable via feature flag: `enable_multi_model_support`
- Extension point: Add new models to `IMAGE_CLASSIFICATION_MODELS`

### Section 2: File & Class Structure

**Model Management:**
```python
IMAGE_CLASSIFICATION_MODELS = [
    'AlexNet', 'BagNet', 'ConvNeXt', 'ConvNeXtTransformer',
    'DarkNet', 'DenseNet', 'EfficientNet', 'FractalNet',
    'GoogLeNet', 'ICNet', 'InceptionV3', 'MNASNet',
    'MaxVit', 'MobileNetV2', 'MobileNetV3', 'RegNet',
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50',
    'ResNet101', 'ShuffleNet', 'SqueezeNet', 'SwinTransformer',
    'VGG', 'VisionTransformer'
]

# Unique NN Functions
unique_nn_cls()       # Filter to classification models
unique_nn()           # Generic filtering for any model list
```

**Functionality:**
- [x] Implement `unique_nn` function
- [x] Implement `unique_nn_cls` function
- [x] List all unique models
- [x] Support ResNet as primary focus
- [ ] Enable additional models (future: feature flag)

### Section 3: Data Generation & Database

**CIFAR-10 Classes (Initial 10):**
```python
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Generate data for classes
generate_class_data_for_epoch(model_name, scheduler_type, epoch, num_classes=10)
```

**Database Integration:**
```python
# Tables
- scheduler_results: Model performance across epochs
- class_data: Per-class accuracy metrics

# Functions
init_database()                          # Create tables and indices
save_scheduler_result()                  # Store epoch results
save_class_accuracy()                    # Store per-class metrics
get_class_accuracies()                   # Retrieve per-class data
```

**Action Items:**
- [x] Generate data for 10 CIFAR classes
- [ ] Extend to 20 classes: `CIFAR10_EXTENDED_CLASSES`
  - Enable via: `enable_extended_classes` feature flag
  - Additional classes defined for future support
- [x] Database functions for data storage
- [x] Class-specific data retrieval

## Database Schema

### scheduler_results Table
```sql
CREATE TABLE scheduler_results (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    scheduler_type TEXT,
    dataset TEXT,
    task TEXT,
    epoch INTEGER,
    accuracy REAL,
    loss REAL,
    best_accuracy INTEGER,
    hyperparameters TEXT (JSON),
    timestamp DATETIME
);
```

### class_data Table
```sql
CREATE TABLE class_data (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    scheduler_type TEXT,
    class_id INTEGER,
    class_name TEXT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    epoch INTEGER,
    timestamp DATETIME
);
```

## Usage Examples

### Example 1: Get Best Classifiers
```python
from ab.gpt.brute.lr.util import unique_nn_cls

# Retrieve best ResNet models for CIFAR-10
best_models = unique_nn_cls(
    epoch_max=90,
    dataset='cifar-10',
    task='img-classification',
    metric='accuracy'
)
print(best_models[['scheduler_type', 'metric_value', 'epoch']])
```

### Example 2: Query Per-Class Metrics
```python
from ab.gpt.brute.lr.util import get_class_accuracies

# Get class-specific accuracy for a model-scheduler combo
class_metrics = get_class_accuracies(
    model_name='ResNet',
    scheduler_type='A0_001',
    epoch=90
)
print(class_metrics)
```

### Example 3: Generate Schedulers
```bash
cd /home/hafsamateen/Project_ResNet
python nn-gpt/ab/gpt/brute/lr/schedulers.py
```

Output:
```
✅ Configuration loaded successfully
✅ Model for fine-tuning: ResNet
✅ Database initialized
✅ Generated 34 scheduler variants
   Generated models: A0_001, A0_002, ..., A0_034
   Classes tracked: 10 (CIFAR-10)
   Database: db/ab.nn.db
```

## Feature Flags

Located in `const.py`:
```python
FEATURES = {
    'enable_huggingface_schedulers': True,      # [x] Enabled
    'enable_multi_model_support': False,        # [ ] Future
    'enable_extended_classes': False,           # [ ] Future (20 classes)
    'enable_database_caching': True,            # [x] Enabled
    'enable_fine_tuning': True,                 # [x] Enabled
}
```

## Future Extensions

### 1. Multi-Model Support
- Set `enable_multi_model_support = True`
- Add models to `ADDITIONAL_MODELS`
- Update fine-tuning logic to handle different architectures

### 2. Extended Classes (20 CIFAR)
- Set `enable_extended_classes = True`
- Define 10 additional classes in `CIFAR10_EXTENDED_CLASSES`
- Update `generate_class_data_for_epoch(num_classes=20)`

### 3. Advanced Fine-Tuning
- Implement curriculum learning
- Add layer-wise learning rate scheduling
- Support mixed precision training

## Testing

### Run Syntax Check
```bash
python -m py_compile ab/gpt/brute/lr/const.py ab/gpt/brute/lr/util.py
```

### Test Imports
```bash
python -c "from ab.gpt.brute.lr import unique_nn_cls, get_active_model"
```

### Test Database Initialization
```bash
python -c "from ab.gpt.brute.lr.util import init_database; init_database()"
```

## Summary

| Requirement | Status | Details |
|------------|--------|---------|
| **Script & Model Configuration** | ✅ Complete | Prefix logic, fine-tuning workflow, ResNet focus |
| **File & Class Structure** | ✅ Complete | 25 classification models, unique_nn functions |
| **Data Generation** | ✅ Complete | 10 CIFAR-10 classes, database integration |
| **Multi-Model Support** | 🔄 Ready | Feature flag `enable_multi_model_support` |
| **Extended Classes (20)** | 🔄 Ready | Feature flag `enable_extended_classes` |
| **Database Functions** | ✅ Complete | Full CRUD operations, indices for performance |

---

**Last Updated:** February 10, 2026  
**Module Version:** 1.0.0
