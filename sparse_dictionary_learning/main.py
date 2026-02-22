import utils.import_dataset as dataset
import bert_embeddings
from utils.config import cfg
import results
from sparse_dictionary import fit_sdl
import os

def run_experiment():
    print(f"--- Starting Analysis: {cfg.dataset_name} ---")

    # Hyperparameters
    n_nonzero_list = [10]
    components_list = [64] #, 64, 128, 256, 512, 1024, 2048]
    layers_to_process = range(12)

    # Load dataset
    df, X = dataset.load()

    # Compute BERT embeddings
    Y_layers = bert_embeddings.compute_embeddings(df, cfg)

    for n_nonzero in n_nonzero_list:
        for n_components in components_list:

            importance_per_layer = {}


            for layer in layers_to_process:
                print(f"\n>>> Processing Layer {layer} | Components {n_components} | Non-zero {n_nonzero_list} <<<")

                # Sparse Dictionary Learning and Probing
                fit_sdl(Y_layers[layer], X, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg)

                log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")

                if os.path.exists(log_path):

                    print("a")
                    # Compute importance per feature
                    #importante_df = results.aggregate_importance(log_path, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg, mode="pct", k=5, pct=0.05)
                    #importance_per_layer[layer] = importante_df

                    # Plot Selectivity Matrix (Atoms vs Features)
                    matrix = results.plot_selectivity_matrix(log_path, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg)

                    # Plot Hierarchical Heatmap based on feature correlations
                    #results.plot_hierarchy_heatmap(matrix, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg)

                    # Plot Identifiability Score distribution
                    #results.plot_identifiability_distribution(log_path, n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg=cfg)

                    # Detailed analysis for specific features or stimuli
                    #results.summarize_feature_probing("animal", log_path=log_path)
                    #results.get_top_stimuli_for_atoms(df, n_components, n_nonzero, layer, cfg, atom_indices=[417, 472])

            #results.plot_importance_across_layers(importance_per_layer, n_components=n_components, n_nonzero=n_nonzero, cfg=cfg)

    print("\n--- Generating global layer-wise identifiability plot ---")
    #results.plot_identifiability(n_layers=len(layers_to_process), n_components=2048, n_nonzero=10, cfg=cfg)
    #results.roc_auc_plot(n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg = cfg)
    #results.p_value_plot(n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg = cfg)
    #results.neff_plot(n_components=n_components, n_nonzero=n_nonzero, layer=layer, cfg = cfg)
    #results.plot_layer_comparison(cfg)
    print("\nDone. All figures are saved in:", cfg.figures_dir)

if __name__ == "__main__":
    run_experiment()