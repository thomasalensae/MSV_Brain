# ============================================================
# Install MLEM
# ============================================================
#pip install -U pip
#pip install "git+https://github.com/LouisJalouzot/MLEM"
#pip install torch transformers numpy pandas scipy scikit-learn matplotlib tqdm umap-learn

from dataclasses import dataclass

@dataclass
class Config:
    dataset_csv: str = "Dataset/relative_clause.csv"  # select the dataset
    sentence_col: str = "sentence" # column that contain the text
    stratify_col: str = "sentence_CLAUSE" # feature to use for stratified sampling
    n_max = 8000

    # Embeddings
    model_name = "bert-base-uncased"
    batch_size = 32
    max_length = 64
    cache_dir = "embeddings_cache"

    # MLEM
    random_seed = 42
    interactions: bool = False
    distance = "euclidean"

    # Visualization (UMAP)
    n_vis = 3000 # subsample for plotting fasteer
    umap_neighbors = 30
    umap_min_dist = 0.15
    umap_metric = "euclidean"

cfg = Config()