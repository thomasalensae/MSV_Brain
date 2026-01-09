import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import cfg
import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from tqdm.auto import tqdm



def set_global_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def stratified_sample(df, by: str, n_total, seed = 42) -> pd.DataFrame:
    """Stratified subsampling keeping class proportions in column `by`
    - by: name of the column used for stratification
    - n_total: desired final size
	"""

    if n_total is None or n_total >= len(df): # no need subsampling
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = df.groupby(by, group_keys=False)
    sizes = (groups.size() / len(df) * n_total).round().astype(int) # number of examples to keep per group

    # Random draw within each group
    sampled_parts = []
    for k, g in groups:
        take = int(sizes.loc[k])
        take = max(1, min(take, len(g)))
        idx = rng.choice(g.index.values, size=take, replace=False)
        sampled_parts.append(df.loc[idx])

    out = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=seed).reset_index(drop=True)
    return out

def build_X_from_df(df: pd.DataFrame, sentence_col: str) -> pd.DataFrame:
    """Build the X matrix for MLEM, excluding the sentence column and casting object columns to categorical"""
    if sentence_col not in df.columns:
        raise ValueError(f"Missing sentence column: {sentence_col}")

    # Remove sentence column (which contains the text)
    X = df.drop(columns=[sentence_col]).copy()

    # Cast object columns to categorical columns for MLEM
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")

    return X


def load():
    set_global_seeds(cfg.random_seed)

    df = pd.read_csv(cfg.dataset_csv)
    print("Loaded:", cfg.dataset_csv)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if cfg.stratify_col not in df.columns:
        raise ValueError(
            f"stratify_col='{cfg.stratify_col}' not found in columns. "
            f"Set cfg.stratify_col to one of: {list(df.columns)}"
        )

    # Subsampling
    df = stratified_sample(df, by=cfg.stratify_col, n_total=cfg.n_max, seed=cfg.random_seed)
    print("After subsample shape:", df.shape)
    print(df[cfg.stratify_col].value_counts())

    # Build X
    X = build_X_from_df(df, sentence_col=cfg.sentence_col)
    print("X shape:", X.shape)
    print(X.dtypes)

    return df, X

