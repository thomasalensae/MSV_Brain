import os
from pathlib import Path
from dataclasses import dataclass

# Base directory for the logic (sparse_dictionary_learning/)
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # 1. Dataset Selection
    dataset_name: str = "relative_clause"
    sentence_col: str = "sentence"
    stratify_col: str = "sentence_CLAUSE"
    n_max: int = 7680  # Updated to match your cached file n_count
    random_seed: int = 40

    # 2. Model Parameters
    model_name: str = "bert-base-uncased"
    batch_size: int = 32
    max_length: int = 64
    extension_model = "_cls"

    def __post_init__(self):

        self.data_dir = BASE_DIR / "data"

        self.dataset_csv = self.data_dir / f"{self.dataset_name}.csv"

        self.base_cache_dir = BASE_DIR / "cache" / f"cache_{self.dataset_name}{self.extension_model}"

        self.embeddings_cache = BASE_DIR / "cache" / f"cache_{self.dataset_name}{self.extension_model}" / "embeddings_cache"
        self.embeddings_cache.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.base_cache_dir / "log"
        self.z_cache = self.base_cache_dir / "Z_cache"
        self.dict_cache = self.base_cache_dir / "sparse_dictionaries_cache"

        self.figures_dir = BASE_DIR / "figures" / f"figures_{self.dataset_name}"

        for path in [self.embeddings_cache, self.log_dir, self.z_cache, self.dict_cache, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)

cfg = Config()