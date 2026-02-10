"""
Learning Rate Scheduler Constants & Configuration Module

This module defines:
- Model configurations for scheduler fine-tuning
- Image classification models list
- Database and prompt parameters
- Scheduler strategy definitions
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Currently active model for fine-tuning
ACTIVE_MODEL = "ResNet"

# Image Classification Models - Core Model List
# These are the primary models supported for LR scheduler experimentation
IMAGE_CLASSIFICATION_MODELS = [
    'AlexNet',
    'BagNet',
    'ConvNeXt',
    'ConvNeXtTransformer',
    'DarkNet',
    'DenseNet',
    'EfficientNet',
    'FractalNet',
    'GoogLeNet',
    'ICNet',
    'InceptionV3',
    'MNASNet',
    'MaxVit',
    'MobileNetV2',
    'MobileNetV3',
    'RegNet',
    'ResNet',           # Primary focus
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ShuffleNet',
    'SqueezeNet',
    'SwinTransformer',
    'VGG',
    'VisionTransformer',
]

# Models planned for future support
ADDITIONAL_MODELS = [
    'AirNet',
    'AirNext',
    'BayesianNet',
    'ComplexNet',
    'DPN68',
    'DPN107',
    'DPN131',
    'Diffuser',
]

# All supported models
ALL_MODELS = IMAGE_CLASSIFICATION_MODELS + ADDITIONAL_MODELS

# ============================================================================
# Learning Rate Scheduler Types
# ============================================================================

SCHEDULER_TYPES = {
    'step_based': [
        'StepLR',
        'MultiStepLR',
        'ExponentialLR',
    ],
    'polynomial': [
        'PolynomialLR_quadratic',
        'PolynomialLR_cubic',
    ],
    'cosine': [
        'CosineAnnealingLR',
        'CosineAnnealingWarmRestarts',
    ],
    'cyclic': [
        'CyclicLR_triangular',
        'CyclicLR_triangular2',
        'CyclicLR_exp_range',
        'OneCycleLR',
    ],
    'warmup': [
        'LinearLR_warmup',
        'ConstantLR_warmup',
        'HF_linear_warmup',
        'HF_cosine_warmup',
    ],
    'lambda': [
        'LambdaLR_linear_decay',
        'LambdaLR_power_decay',
    ],
    'multiplicative': [
        'MultiplicativeLR_exponential',
    ],
}

# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'path': 'db/ab.nn.db',
    'use_cache': True,
    'auto_refresh': True,
}

# ============================================================================
# CIFAR-10 Dataset Classes Configuration
# ============================================================================

# Standard CIFAR-10 classes (initial 10 classes)
CIFAR10_CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

CIFAR10_CLASSES_COUNT = len(CIFAR10_CLASSES)

# Extended classes for future support (remaining 10 classes for CIFAR-20 concept)
CIFAR10_EXTENDED_CLASSES = [
    'aircraft',
    'bus',
    'train',
    'boat',
    'motorcycle',
    'bicycle',
    'animal',
    'plant',
    'building',
    'vehicle',
]

# ============================================================================
# Prompt Configuration
# ============================================================================

PROMPT_CONFIG = {
    'model_detection': {
        'prefix': 'model_',
        'active_indicator': '<<<ACTIVE>>>',
        'keywords': ['ResNet', 'checkpoint', 'fine-tune'],
    },
    'lr_extraction': {
        'keywords': ['learning_rate', 'lr', 'learning rate'],
        'default_lr': 0.1,
        'lr_range': [0.001, 1.0],
    },
    'database': {
        'query_prefix': 'SELECT * FROM schedulers WHERE',
        'storage_prefix': 'INSERT INTO schedulers',
    },
}

# ============================================================================
# Hyperparameter Defaults
# ============================================================================

DEFAULT_HYPERPARAMS = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'epoch_max': 90,
    'batch_size': 128,
    'warmup_epochs': 0.1,
    'step_size': 30,
    'gamma': 0.1,
    'T_0': 10,
    'T_mult': 2,
    'eta_min': 0.0,
    'min_lr': 0.01,
}

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIRS = {
    'base': 'out/nngpt/llm/epoch/A0/synth_nn',
    'models': 'out/nngpt/llm/epoch/A0/synth_nn/models',
    'results': 'out/nngpt/llm/epoch/A0/results',
    'logs': 'out/nngpt/llm/epoch/A0/logs',
}

# ============================================================================
# File Naming Conventions
# ============================================================================

FILE_CONVENTIONS = {
    'model_prefix': 'A0_',
    'model_file': 'new_nn.py',
    'hp_file': 'hp.txt',
    'results_file': 'results.json',
    'log_file': 'training.log',
}

# ============================================================================
# Feature Flags
# ============================================================================

FEATURES = {
    'enable_huggingface_schedulers': True,
    'enable_multi_model_support': False,  # To be enabled when adding more models
    'enable_extended_classes': False,     # To be enabled when supporting 20 classes
    'enable_database_caching': True,
    'enable_fine_tuning': True,
}
