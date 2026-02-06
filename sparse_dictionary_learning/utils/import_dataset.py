import random
import numpy as np
import pandas as pd
from sparse_dictionary_learning.utils.config import cfg
import torch

from sklearn.preprocessing import OneHotEncoder


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

    # Subsampling (not needed if dataset is small enough)
    df = stratified_sample(df, by=cfg.stratify_col, n_total=cfg.n_max, seed=cfg.random_seed)
    print("After subsample shape:", df.shape)
    print(df[cfg.stratify_col].value_counts())


    # Build X (remove text column, cast object columns to categorical)
    X = build_X_from_df(df, sentence_col=cfg.sentence_col)
    print("X shape:", X.shape)

    X = pd.get_dummies(X)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_enc = encoder.fit_transform(X)

    feature_names = encoder.get_feature_names_out(X.columns)
    keep_indices = [i for i, name in enumerate(feature_names) if "False" not in name]
    X_filtered = X_enc[:, keep_indices]
    names_filtered = feature_names[keep_indices]
    new_feature_names = np.array([name.replace("_True", "") for name in names_filtered])
    print("Features: \n" + new_feature_names.tolist().__str__())

    X_ohe = pd.DataFrame(X_filtered, columns=new_feature_names, index=X.index)

    return df, X

