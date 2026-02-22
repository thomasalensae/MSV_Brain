import random
import numpy as np
import pandas as pd
from sparse_dictionary_learning.utils.config import cfg
import torch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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
    """Build the X matrix, excluding the sentence column and casting object columns to categorical"""
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
    #df = stratified_sample(df, by=cfg.stratify_col, n_total=cfg.n_max, seed=cfg.random_seed)
    #print("After subsample shape:", df.shape)
    #print(df[cfg.stratify_col].value_counts())


    X = build_X_from_df(df, sentence_col=cfg.sentence_col)

    zipf_cols = [c for c in X.columns if "ZIPF" in c]
    other_cols = [c for c in X.columns if c not in zipf_cols]

    categ_cols = X[other_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_nonzipf_cols = [c for c in other_cols if c not in categ_cols]

    if len(zipf_cols) > 0:
        X[zipf_cols] = X[zipf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        for col in zipf_cols:
            if col != "verb_ZIPF":
                X[col] = X[col].round().astype(int)

    all_categ_cols = categ_cols + zipf_cols

    # Utilisation de drop="first" pour éviter d'avoir les colonnes inverses
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop="first"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_and_zipf", ohe, all_categ_cols),
            ("num", "passthrough", numeric_nonzipf_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    Xt = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_final = pd.DataFrame(Xt, columns=feature_names, index=X.index)
    cleaned_columns = [c.rpartition('_')[0] if '_' in c else c for c in X_final.columns]
    X_final.columns = cleaned_columns
    print(f"ZIPF passthrough: {zipf_cols}")
    print(f"Categorical OHE: {categ_cols}")

    if len(numeric_nonzipf_cols) > 0:
        print(f"Other numeric passthrough: {numeric_nonzipf_cols}")
    print("Final features:", X_final.shape[1])

    return df, X_final


