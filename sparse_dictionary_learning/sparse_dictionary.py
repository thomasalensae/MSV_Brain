import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
import time
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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

def fit_sdl_cv(Y_layer, X, n_components, n_nonzero, layer, cfg, n_splits=5, random_state=42, n_rep=20):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y_layer)

    for fold, (train_idx, test_idx) in enumerate(kf.split(Y_std)):
        print(f"\n>>> Fold {fold} <<<")
        start_time_fold = time.time()

        log_path = os.path.join(cfg.log_dir_cv, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}_fold{fold}.jsonl")
        z_train_path = os.path.join(cfg.z_cache_cv, f"Z_train_L{layer}_C{n_components}_K{n_nonzero}_fold{fold}.npy")
        z_test_path = os.path.join(cfg.z_cache_cv, f"Z_test_L{layer}_C{n_components}_K{n_nonzero}_fold{fold}.npy")

        if os.path.exists(log_path):
            continue

        if os.path.exists(z_train_path) and os.path.exists(z_test_path):
            print(f"Loading cached Z codes for fold {fold}")
            Z_train = np.load(z_train_path)
            Z_test = np.load(z_test_path)
        else:
            print(f"Learning dictionary for fold {fold} on {len(train_idx)} samples...")
            start_time_dict = time.time()
            dict_learner = MiniBatchDictionaryLearning(
                n_components=n_components,
                transform_algorithm="omp",
                transform_n_nonzero_coefs=n_nonzero,
                batch_size=256,
                random_state=random_state
            )

            Z_train = dict_learner.fit_transform(Y_std[train_idx])
            Z_test = dict_learner.transform(Y_std[test_idx])

            np.save(z_train_path, Z_train)
            np.save(z_test_path, Z_test)
            print(f"Time SDL Fold {fold}: {time.time() - start_time_dict:.2f}s")

        for i, feature_name in enumerate(X.columns):
            start_time_feature = time.time()

            y_train = X.iloc[train_idx, i].values
            y_test = X.iloc[test_idx, i].values

            clf = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                C=1.0,
                n_jobs=-1,
                random_state=random_state
            )
            clf.fit(Z_train, y_train)

            y_pred = clf.predict(Z_test)
            coefs = clf.coef_.ravel()

            auc_hat = float(roc_auc_score(y_test, y_pred))

            w_m = compute_atom_importance_manual(clf, Z_test, y_test, random_state, n_rep)

            p_value = compute_feature_pvalue(y_test, y_pred, auc_hat, n_repeats=n_rep, random_state=random_state)

            all_importances = {int(a): float(w_m[a]) for a in range(len(w_m))}
            all_coefs = {int(a): float(coefs[a]) for a in range(len(coefs))}

            log_entry = {
                "feature": feature_name,
                "layer": layer,
                "n_components": n_components,
                "n_nonzero": n_nonzero,
                "roc_auc": auc_hat,
                "p_value": p_value,
                "permutation_importance": all_importances,
                "coef": all_coefs,
                "timestamp": datetime.now().isoformat()
            }

            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

        print(f"Total time fold {fold}: {time.time() - start_time_fold:.2f}s")

def fit_sdl(Y_layer, X, n_components, n_nonzero, layer, cfg, test_size=0.2, random_state=42, top_k_atoms=30, n_perm_repeats=20, n_bootstraps=20):
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
            batch_size=512,
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

        all_importances = {int(a): float(w_m[a]) for a in range(len(w_m))}
        all_coefs = {int(a): float(coefs[a]) for a in range(len(coefs))}

        log_entry = {
                "feature": feature_name,
                "layer": layer,
                "n_components": n_components,
                "n_nonzero": n_nonzero,
                "roc_auc": auc_hat,
                "p_value": p_value,
                "permutation_importance": all_importances,
                "coef": all_coefs,
                "timestamp": datetime.now().isoformat()
            }

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

