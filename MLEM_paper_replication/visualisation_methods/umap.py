import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dataset.config import cfg
from utility_functions import stratified_indices, labeling

def run_umap_2d(Y,seed = 42,n_neighbors = 30,min_dist = 0.15,metric = "euclidean"):
    """UMAP 2D projection."""
    reducer = umap.UMAP(n_components=2,n_neighbors=n_neighbors,min_dist=min_dist, metric=metric, random_state=seed)
    return reducer.fit_transform(Y)


def plot_umap_grid(coords_layers: list[np.ndarray], y_color: np.ndarray, class_names: list[str], highlight_layer_idx: int | None = None, save_path: str | None = None, point_size: int = 5,) -> None:
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


def visualisation(df, Y_layers):

    Y_layers_vis, y_vis, class_names , n_vis = labeling(df, Y_layers)

    # UMAP per layer
    coords_layers = []
    for li in range(12):
        print(f"UMAP layer {li+1}/12 on n={n_vis}")
        coords_layers.append(
            run_umap_2d(
                Y_layers_vis[li],
                seed=cfg.random_seed,
                n_neighbors=cfg.umap_neighbors,
                min_dist=cfg.umap_min_dist,
                metric=cfg.umap_metric,
            )
        )

    plot_umap_grid(
        coords_layers=coords_layers,
        y_color=y_vis,
        class_names=class_names,
        highlight_layer_idx=cfg.peak_layer_idx,
        save_path="figure5_umap.png",
        point_size=5,
    )
