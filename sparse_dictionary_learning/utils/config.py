
from dataclasses import dataclass

@dataclass
class Config:
    dataset_name: str = "relative_clause"
    dataset_csv: str = f"/Users/maxime/MSV_Brain/sparse_dictionary_learning/data/{dataset_name}.csv"
    sentence_col: str = "sentence" # column that contain the text
    stratify_col: str = "sentence_CLAUSE" # feature to use for stratified sampling
    n_max = 8000
    random_seed=40

    # Embeddings
    model_name = "bert-base-uncased"
    batch_size = 32
    max_length = 64
    cache_dir = f"/Users/maxime/MSV_Brain/sparse_dictionary_learning/cache_{dataset_name}/embeddings_cache"


cfg = Config()