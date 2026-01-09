from sklearn.manifold import MDS
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from dataset.config import cfg
from utility_functions import stratified_indices, labeling

def run_mds_2d(Y, random_state = 42, metric = True,n_init = 1,max_iter = 80):
    """MDS 2D projection (tune n_init/max_iter for stability vs speed)."""
    mds = MDS(n_components=2, metric=metric, dissimilarity="euclidean", random_state=random_state, n_init=n_init, max_iter=max_iter, normalized_stress="auto")

    return mds.fit_transform(Y)

def plot_mds_grid(coords_layers: list[np.ndarray],y_color: np.ndarray,class_names: list[str],highlight_layer_idx: int | None = None,save_path: str | None = None, point_size: int = 6):
    """Plot a 3x4 grid for 12 layers."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for i in range(12):
        ax = axes[i]
        coords = coords_layers[i]

        for c in np.unique(y_color):
            idx = (y_color == c)
            ax.scatter(coords[idx, 0], coords[idx, 1], s=point_size, alpha=0.8, label=class_names[c])

        ax.set_title(f"Layer {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

        if highlight_layer_idx is not None and i == highlight_layer_idx:
            for spine in ax.spines.values():
                spine.set_linewidth(3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(class_names), frameon=False)

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print("Saved figure:", save_path)

    plt.show()

def visualisation(df, Y_layers, fast = False):
    # Labels for coloring: use the same column as stratification by default

    Y_layers, y, class_names, n = labeling(df, Y_layers,fast=fast)

    # MDS per layer
    coords_layers_mds = []
    for li in range(12):
        print(f"MDS (fast) layer {li+1}/12 on n={n}")
        coords_layers_mds.append(
            run_mds_2d(
                Y_layers[li],
                random_state=cfg.random_seed,
                n_init=1,
                max_iter=80,
                metric=True,
            )
        )

    plot_mds_grid(
        coords_layers=coords_layers_mds,
        y_color=y,
        class_names=class_names,
        highlight_layer_idx=cfg.peak_layer_idx,
        save_path="figure5_mds.png",
        point_size=6,
    )
