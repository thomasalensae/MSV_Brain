import numpy as np
from dataset.config import cfg
from sklearn.preprocessing import LabelEncoder
def stratified_indices(y, n_total, seed = 42):
    """select a subset of indices while preserving class proportions in y"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    take_per_class = np.maximum(1, np.round(props * n_total).astype(int))

    idx_all = []
    for c, take in zip(classes, take_per_class):
        idx_c = np.where(y == c)[0]
        take = min(take, len(idx_c))
        idx_all.append(rng.choice(idx_c, size=take, replace=False))

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    if len(idx) > n_total:
        idx = idx[:n_total]
    return idx

def labeling(df, Y_layers, fast = False):
    # Labels for coloring: use the same column as stratification by default
    label_col = cfg.stratify_col
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).values)
    class_names = list(le.classes_)

    n = len(df)

    if fast:
        n = min(cfg.n_vis, len(df))
        vis_idx = stratified_indices(y, n_total=n, seed=cfg.random_seed)

        Y_layers = [Y[vis_idx] for Y in Y_layers]
        y = y[vis_idx]

    return Y_layers, y, class_names, n