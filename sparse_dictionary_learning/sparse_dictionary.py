import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_squared_error, roc_auc_score
from sklearn.inspection import permutation_importance

def setup_custom_logger(log_file):
    logger = logging.getLogger("LinguisticProbing")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger

def is_binary(y):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    uniq = np.unique(y)
    return len(uniq) == 2 and set(uniq).issubset({0, 1})

def fit_sdl(Y_layer, X, n_components, n_nonzero, layer, cfg, test_size=0.2, random_state=42, top_k_atoms=30, n_perm_repeats=10, n_bootstraps=50):
    """
    Fits Sparse Dictionary Learning
    """

    log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")
    z_path = os.path.join(cfg.z_cache, f"Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy")

    if os.path.exists(log_path):
        print(f"Skipping: log already exists at {log_path}")
        return

    logger = setup_custom_logger(log_path)
    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y_layer)

    # Sparse Dictionary Learning (Z)
    if os.path.exists(z_path):
        print(f"Loading cached Z codes from: {z_path}")
        Z = np.load(z_path)
    else:
        print(f"Learning dictionary for layer {layer}...")
        dict_learner = MiniBatchDictionaryLearning(
            n_components=n_components,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=n_nonzero,
            batch_size=256,
            random_state=random_state
        )
        Z = dict_learner.fit_transform(Y_std)
        np.save(z_path, Z)
        print(f"Saved Z codes to: {z_path}")

    # Split Train/Test
    X_train_indices, X_test_indices = train_test_split(np.arange(len(X)), test_size=test_size, random_state=random_state)

    Z_train, Z_test = Z[X_train_indices], Z[X_test_indices]

    # Iterate over each linguistic feature
    for i, feature_name in enumerate(X.columns):
        start_time = time.time()

        # Retrieve binary labels for feature j
        y_train = X.iloc[X_train_indices, i].values
        y_test = X.iloc[X_test_indices, i].values

        print(f"Evaluating feature: {feature_name}")

        # Logistic Regression (Probing)
        clf = LogisticRegression(
            penalty="l1",      # Lasso penalty as requested
            solver="saga",     # Required for L1
            max_iter=5000,
            C=1.0,
            n_jobs=-1,
            random_state=random_state
        )
        clf.fit(Z_train, y_train)

        # Performance
        y_pred = clf.predict(Z_test)
        coefs = clf.coef_.ravel()

        auc_hat = float(roc_auc_score(y_test, y_pred))

        # w_m importances via permutation
        w_m = compute_atom_importance_manual(clf, Z_test, y_test, random_state, 100)

        p_value = compute_feature_pvalue(y_test, y_pred, auc_hat, n_repeats=100, random_state=random_state)

        # Identify Top Atoms (based on permutation)
        top_atoms_indices = np.argsort(w_m)[::-1][:top_k_atoms]

        log_entry = {
            "feature": feature_name,
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "n_components": n_components,
            "n_nonzero": n_nonzero,

            "roc_auc": auc_hat,
            "p_value": p_value,

            "top_atoms": top_atoms_indices.tolist(),
            "permutation_importance": {int(a): float(w_m[a]) for a in top_atoms_indices},
        }

        print(f"  AUC: {auc_hat:.3f} | Time: {time.time() - start_time:.2f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")


def compute_atom_importance_manual(clf, Z_test, y_test, random_state, n_repeats=10):
    """
    Calculates the importance w_m of each atom via permutation.

    w_m = AUC_initial - mean(AUC_perm)
    """

    y_prob_base = clf.predict_proba(Z_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_base) # AUC^

    n_samples, n_atoms = Z_test.shape
    w = np.zeros(n_atoms)

    rng = np.random.RandomState(random_state)
    Z_work = Z_test.copy()

    for m in range(n_atoms):

        if clf.coef_[0, m] == 0:
            w[m] = 0.0
            continue

        auc_perms = []
        original_column = Z_work[:, m].copy()

        # 3. Repeat B times (Permutation loop)
        for b in range(n_repeats):

            rng.shuffle(Z_work[:, m])

            y_prob_perm = clf.predict_proba(Z_work)[:, 1]
            score_perm = roc_auc_score(y_test, y_prob_perm)
            auc_perms.append(score_perm) # AUC_perm

        Z_work[:, m] = original_column

        # w_m = AUC^ - (1/B) * Sum(AUC_perm)
        mean_perm_auc = np.mean(auc_perms)
        w[m] = roc_auc - mean_perm_auc

    return w

def compute_feature_pvalue(y_true, y_score, score_observed, n_repeats=100, random_state=42):

    rng = np.random.RandomState(random_state)
    null_scores = []

    y_shuffled = y_true.copy()

    for _ in range(n_repeats):
        rng.shuffle(y_shuffled)
        score_null = roc_auc_score(y_shuffled, y_score)
        null_scores.append(score_null)

    null_scores = np.array(null_scores)

    n_greater = np.sum(null_scores >= score_observed)
    p_value = (n_greater + 1) / (n_repeats + 1)

    return p_value

def compute_neff(w_m):
    """
    Calculates the Effective Number of Atoms (N_eff) based on Hill's entropy of order 2.

    1. Normalization: p_m = w_m / sum(w)
    2. N_eff = 1 / sum(p_m^2)
    """
    w_pos = np.maximum(w_m, 0)
    sum_w = np.sum(w_pos)

    if sum_w == 0:
        return 0.0, np.zeros_like(w_pos)

    # Probability distribution p_m
    p_m = w_pos / sum_w

    # Effective number N_eff
    n_eff = 1.0 / np.sum(p_m ** 2)

    return n_eff, p_m

