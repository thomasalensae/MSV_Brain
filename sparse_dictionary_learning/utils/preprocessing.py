import numpy as np
from sklearn.preprocessing import normalize

def preprocess_embeddings(
    Y: np.ndarray,
    center: bool = True,
    l2_normalize: bool = False,
) -> np.ndarray:
    """
    Preprocess embeddings before dictionary learning.

    Parameters
    ----------
    Y : array of shape (n_samples, n_features)
    center : whether to subtract the mean embedding
    l2_normalize : whether to normalize each sample to unit norm

    Returns
    -------
    Y_proc : preprocessed embeddings
    """
    Y_proc = Y.copy()

    if center:
        Y_proc -= Y_proc.mean(axis=0, keepdims=True)

    if l2_normalize:
        Y_proc = normalize(Y_proc, axis=1)

    return Y_proc