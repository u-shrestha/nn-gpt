# config/config.py

import os

from ab.nn.util.Const import out_dir

# Base directories (adjust as needed)
DATASET_DESC_DIR = os.path.join(out_dir, "dataset_descriptions")
GITHUB_REPO_DIR = os.path.join(out_dir, "github_repos")
FAISS_INDEX_PATH = os.path.join(out_dir, "index", "faiss_index.index")
FINE_TUNED_MODEL_DIR = os.path.join(out_dir, "fine_tuned_model")

CODE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 8
TOP_K_RETRIEVAL = 5

HF_TOKEN = "YOUR_HF_ACCESS_TOKEN_HERE"  
