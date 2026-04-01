import utils.import_dataset as dataset
import bert_embeddings
from utils.config import cfg
import results
from sparse_dictionary import fit_sdl, fit_sdl_cv
import os

def run_experiment():
    print(f"--- Starting Analysis: {cfg.dataset_name} ---")

    # Hyperparameters
    n_nonzero_list = [10]
    components_list = [64]
    layers_to_process = range(12) # Process all layers from 0 to 12
    n_splits = 5

    # Load dataset
    df, X = dataset.load()

    # Compute BERT embeddings
    Y_layers = bert_embeddings.compute_embeddings(df, cfg)

    for n_nonzero in n_nonzero_list:
        for n_components in components_list:

            importance_per_layer = {}

            for layer in layers_to_process:
                print(f"\n>>> Processing Layer {layer} | Components {n_components} | Non-zero {n_nonzero_list} <<<")

                log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")

                # Sparse Dictionary Learning and Probing
                fit_sdl_cv(Y_layers[layer], X, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg, n_splits=n_splits)
                fit_sdl(Y_layers[layer], X, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg)


                # Plot Selectivity Matrix (Atoms vs Features)
                if layer == 5 or layer == 11:
                    matrix = results.plot_matrix(log_path, n_components, n_nonzero, layer, cfg)
                    matrix = results.plot_importance_matrix(log_path, n_components, n_nonzero, layer, cfg)

                #results.plot_conditional_heatmap(X, layer, n_components, n_nonzero, "sentence_RC_attached", cfg, top_k=10)
                #results.plot_conditional_heatmap(X, layer, n_components, n_nonzero, "subj_NUM", cfg, top_k=10)


            print("\n--- Layer-wise plot ---")

            print(" -- Roc AUC Plot -- ")
            results.roc_auc_plot(n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg = cfg)

            print(" -- Neff Plot -- ")
            results.neff_plot_cv(n_components=n_components, n_nonzero=n_nonzero, cfg = cfg)

            print(" -- Elbow Plot -- ")
            results.plot_elbow(n_components, n_nonzero, cfg, layers = [0, 5, 11])
            results.plot_elbow_comparaison(cfg, layers = [0, 5, 11])

            print(" -- Distribution Plot -- ")
            results.plot_multi_layer_distributions(X, [0, 5, 11], n_components, n_nonzero, cfg)

            print("\nDone. All figures are saved in:", cfg.figures_dir)

if __name__ == "__main__":
    run_experiment()