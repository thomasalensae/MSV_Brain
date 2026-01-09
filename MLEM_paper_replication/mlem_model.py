import pandas as pd
import numpy as np
from dataset.config import cfg
from dataset.import_dataset import stratified_sample, build_X_from_df
from mlem import MLEM
import matplotlib.pyplot as plt


def run_mlem_across_layers(X: pd.DataFrame, Y_layers: list[np.ndarray], random_seed = 42, interactions: bool = False, distance: str = "euclidean",) -> tuple[list[MLEM], list[pd.DataFrame], list[dict]]:
    """Fit one MLEM model per layer and return models, feature-importance tables, and raw scores."""
    models = []
    fi_tables = []
    score_objs = []

    for li in range(len(Y_layers)):
        print(f"Training MLEM on layer {li+1}/{len(Y_layers)}")
        Y = Y_layers[li]

        m = MLEM(interactions=interactions, random_seed=random_seed, distance=distance,)
        m.fit(X, Y)

        fi, scores = m.score()
        models.append(m)
        fi_tables.append(fi)
        score_objs.append(scores)

    return models, fi_tables, score_objs


def plot_feature_importance_curves(importance_df, top_k=None, features=None, normalize=False, figsize=(8, 3), title="MLEM feature importance across layers"):
    """ Plot feature-importance curves across layers """
    df = importance_df.copy()

    try:
        layer_nums = df.index.to_series().astype(str).str.extract(r"(\d+)")[0].astype(int).values
    except Exception:
        layer_nums = list(range(1, len(df) + 1))

    # Select features
    if features is not None:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Requested features not found in importance_df: {missing}")
        df = df[features]
    elif top_k is not None:
        top = df.mean(axis=0).sort_values(ascending=False).head(top_k).index.tolist()
        df = df[top]

    if normalize:
        denom = df.max(axis=0).replace(0, 1e-12)
        df = df / denom

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        ax.plot(layer_nums, df[col].values, linewidth=2, markersize=3, label=col)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.set_xticks(sorted(set(layer_nums)))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    return fig, ax

def display_feature_importance(mlem_fi_tables, top_k = 10, listed_features = None):
    """Display top_k features from a feature-importance table."""

    # Summarize mean importance per feature per layer
    mean_importance_rows = []
    for li, fi in enumerate(mlem_fi_tables):
        row = fi.mean(axis=0)
        row.name = f"layer_{li+1}"
        mean_importance_rows.append(row)

    importance_df = pd.concat(mean_importance_rows, axis=1).T
    print(importance_df)

    feature_of_interest = cfg.stratify_col

    if feature_of_interest not in importance_df.columns:
        raise ValueError(
            f"feature_of_interest='{feature_of_interest}' not in X columns. "
            f"Available: {list(importance_df.columns)}"
        )

    peak_layer_idx = int(np.argmax(importance_df[feature_of_interest].values))  # 0..11
    cfg.peak_layer_idx = peak_layer_idx
    print("Feature of interest:", feature_of_interest)
    print("Peak layer (1..12):", peak_layer_idx + 1)

    print("\nTop layers for this feature:")
    print(importance_df[[feature_of_interest]].sort_values(feature_of_interest, ascending=False).head(8))

    # Plot the importance curves for all features
    plot_feature_importance_curves(importance_df)

    # Top 5 variables the most important in average
    plot_feature_importance_curves(importance_df, top_k=top_k)

    if listed_features is not None:
        listed_features = ["sentence_CLAUSE", "sentence_RC_attached", "subj_NUM", "obj_NUM"]

    plot_feature_importance_curves(importance_df, features=listed_features)
