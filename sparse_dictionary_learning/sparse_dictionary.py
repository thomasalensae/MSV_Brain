import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from utils.preprocessing import preprocess_embeddings
import os


def setup_custom_logger(log_file="experiment_log.jsonl"):
    logger = logging.getLogger("LinguisticProbing")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    return logger

def fit_sdl(Y_layer, X, n_components=512, n_nonzero=20, layer = 6,test_size=0.2, random_state=0, use_saved_Z=True, Z_path="Z.npy", top_k_atoms=10, n_perm_repeats=10):

    log_name = f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl"
    print(log_name)

    if os.path.exists(log_name):
        return
    else:
        logger = setup_custom_logger(log_name)

    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y_layer)

    # Dictionary learning
    dict_learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=n_nonzero,
        batch_size=256,
        random_state=random_state
    )

    Z_path=f"sparse_dictionary_learning/cache/Z_cache/Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"

    if os.path.exists(Z_path):
        print("Loading sparse codes Z")
        Z = np.load(Z_path)
    else:
        print("Learning dictionary and computing sparse codes Z")
        Z = dict_learner.fit_transform(Y_std)
        np.save(Z_path, Z)

    # Split
    X_array = X.values
    X_train, X_test, Z_train, Z_test = train_test_split(X_array, Z, test_size=test_size, random_state=random_state)

    # Probing feature per feature
    start_index = 0

    for i, feature_name in enumerate(X.columns[start_index:], start=start_index):

        start_time = time.time()

        y_train = X_train[:, i]
        y_test = X_test[:, i]

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        print("Evaluating feature:", feature_name)

        clf = LogisticRegression(penalty="l1", solver="saga", max_iter=1000, C=1.0, n_jobs=-1, random_state=random_state)

        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        acc = accuracy_score(y_test, y_pred)

        # Classifier weight: most important atoms (|coef|)
        coefs = clf.coef_.ravel()
        top_atoms = np.argsort(np.abs(coefs))[::-1][:top_k_atoms]
        top_atoms_str = ",".join([f"{a}:{coefs[a]:.4g}" for a in top_atoms])

        # Permutation importance (per atom) for this feature
        perm = permutation_importance(
            clf,
            Z_test,
            y_test,
            n_repeats=n_perm_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring="accuracy"
        )

        perm_mean = perm.importances_mean
        top_perm_atoms = np.argsort(perm_mean)[::-1][:top_k_atoms]
        top_perm_atoms_str = ",".join([f"{a}:{perm_mean[a]:.4g}" for a in top_perm_atoms])

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Top atoms by weight: {top_atoms_str}")
        print(f"  Top atoms by permutation importance: {top_perm_atoms_str}")

        end_time = time.time()
        print(f"Time taken : {end_time - start_time:.2f} s\n")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "feature": feature_name,
            "n_components": n_components,
            "n_nonzero": n_nonzero,
            "accuracy": acc,
            "weights": {int(a): float(coefs[a]) for a in top_atoms},
            "permutation_importance": {int(a): float(perm_mean[a]) for a in top_perm_atoms}
        }
        logger.info(json.dumps(log_entry))

