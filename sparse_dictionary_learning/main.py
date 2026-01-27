import utils.import_dataset as dataset
import bert_embeddings
import utils.config as config
import results
from sparse_dictionary import fit_sdl

print("Starting ...")

# Parameters
n_components=512
n_nonzero=10

#for layer in [0,1,2,3,4,5,6,7,8,9,10,11]:
"""

for layer in [8, 9,10,11]:
    print("Processing layer:", layer)
    # Load dataset
    df, X = dataset.load()

    # BERT embeddings
    Y_layers = bert_embeddings.compute_embeddings(df, config.cfg)
    Y_layer = Y_layers[layer]

    # Fit SDL and probe features
    fit_sdl(Y_layer, X, n_components=n_components, n_nonzero=n_nonzero, layer = layer)

    # Plot results
    #results.summarize_feature_probing("sentence_CLAUSE_objwho", log_path=f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")
    #results.plot_atom_importance("sentence_CLAUSE_objwho", log_path=f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")
    #top_examples = results.get_top_stimuli_for_atoms(df, n_components=n_components, n_nonzero=n_nonzero, layer = layer, atom_indices=[417, 472])

    results.plot_selectivity_matrix(f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl", n_components=n_components, n_nonzero=n_nonzero, layer = layer)

    # Criteria of identifiability

    results.plot_identifiability_distribution(log_path=f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl", n_components=n_components, n_nonzero=n_nonzero, layer = layer)

"""
results.plot_identifiability(12, n_components,n_nonzero)
print("Done.")