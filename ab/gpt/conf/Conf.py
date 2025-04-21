from ab.gpt.util.Const import nnrag_dir

# Base directories (adjust as needed)
DATASET_DESC_DIR = nnrag_dir / 'dataset_descriptions'
GITHUB_REPO_DIR = nnrag_dir / 'github_repos'
FINE_TUNED_MODEL_DIR = nnrag_dir / 'fine_tuned_model'
FAISS_INDEX_PATH = str(nnrag_dir / 'index' / 'faiss_index.index')

CODE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 8
TOP_K_RETRIEVAL = 5

HF_TOKEN = "YOUR_HF_ACCESS_TOKEN_HERE"  
