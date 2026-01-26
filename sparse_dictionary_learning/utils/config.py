
from dataclasses import dataclass

@dataclass
class Config:
    dataset_csv: str = "sparse_dictionary_learning/data/relative_clause.csv"  # select the dataset
    sentence_col: str = "sentence" # column that contain the text
    stratify_col: str = "sentence_CLAUSE" # feature to use for stratified sampling
    n_max = 8000
    random_seed=40

    # Embeddings
    model_name = "bert-base-uncased"
    batch_size = 32
    max_length = 64
    cache_dir = "sparse_dictionary_learning/cache/embeddings_cache"


cfg = Config()