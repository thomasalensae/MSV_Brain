import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_atom_importance(feature_name, log_path="experiment_log.jsonl", top_k=10):

    with open(log_path, 'r') as f:
        data = [json.loads(line) for line in f if json.loads(line)["feature"] == feature_name]
    if not data: return
    entry = data[0]

    atoms = list(entry["permutation_importance"].keys())
    perm_vals = [entry["permutation_importance"][a] for a in atoms]
    weight_vals = [entry["weights"].get(a, 0) for a in atoms]

    idx = np.argsort(perm_vals)
    atoms = [atoms[i] for i in idx]
    perm_vals = [perm_vals[i] for i in idx]
    weight_vals = [weight_vals[i] for i in idx]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_perm = 'skyblue'
    ax1.set_xlabel('Importance by Permutation (Decrease in Accuracy)', color='blue')
    ax1.barh(np.arange(len(atoms)) + 0.2, perm_vals, 0.4, label='Permutation', color=color_perm)
    ax1.tick_params(axis='x', labelcolor='blue')
    ax1.set_yticks(np.arange(len(atoms)))
    ax1.set_yticklabels([f"Atom {a}" for a in atoms])

    ax2 = ax1.twiny()
    ax2.set_xlabel('Magnitude of weight (Coeff L1)', color='red')
    ax2.barh(np.arange(len(atoms)) - 0.2, [abs(w) for w in weight_vals], 0.4, label='weight (abs)', color='salmon', alpha=0.6)
    ax2.tick_params(axis='x', labelcolor='red')

    plt.title(f"Contribution of Atoms to the feature : {feature_name}\n(Acc: {entry['accuracy']:.2%})")
    fig.tight_layout()
    plt.show()

def summarize_feature_probing(feature_name, log_path="experiment_log.jsonl"):

    feature_data = None
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry["feature"] == feature_name:
                feature_data = entry
                break

    if not feature_data:
        print(f"Feature '{feature_name}' not found in logs.")
        return

    print(f"--- ANALYSE OF PROBING : {feature_name} ---")
    print(f"Accuracy of the classifier : {feature_data['accuracy']:.4%}")

    weights = feature_data["weights"]
    perms = feature_data["permutation_importance"]

    top_weight_atoms = set(weights.keys())
    top_perm_atoms = set(perms.keys())
    consensus_atoms = top_weight_atoms.intersection(top_perm_atoms)

    print(f"\nAreas of consensus (present in both analyses) : {list(consensus_atoms)}")

    print("\nDetails of Top Atoms (Weight) :")
    for atom, val in weights.items():
        role = "Activation" if val > 0 else "Suppression"
        print(f"  - Atom {atom}: {val:.4f} ({role})")

def get_top_stimuli_for_atoms(sentences_df, n_components=512, n_nonzero=20, layer = 6, atom_indices = [0, 1], top_k=10):
    """
    Extracts the sentences that most strongly activate specific atoms.
    Z: Sparse code matrix (n_sentences, n_components)
    sentences_df: DataFrame containing the original sentences (must be aligned with Z)
    atom_indices: List of atom indices to analyze
    """

    Z_path=f"cache/Z_cache/Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"
    if os.path.exists(Z_path):
        print("Loading sparse codes Z")
        Z = np.load(Z_path)
    else:
        raise FileNotFoundError(f"Sparse codes Z not found at {Z_path}. Please compute them first.")
    results = {}

    if isinstance(sentences_df, pd.DataFrame):
        text_col = 'sentence' if 'sentence' in sentences_df.columns else sentences_df.columns[0]
        texts = sentences_df[text_col].values
    else:
        texts = sentences_df

    for atom_idx in atom_indices:
        activations = Z[:, atom_idx]
        top_indices = np.argsort(activations)[::-1][:top_k]

        print(f"\n=== TOP STIMULI FOR THE ATOM {atom_idx} ===")
        print(f"(Maximum activations between {activations[top_indices[0]]:.4f} and {activations[top_indices[-1]]:.4f})")

        exemplars = []
        for rank, idx in enumerate(top_indices):
            sentence = texts[idx]
            act_val = activations[idx]
            print(f"{rank+1}. [{act_val:.4f}] {sentence}")
            exemplars.append((sentence, act_val))

        results[atom_idx] = exemplars

    return results

def plot_selectivity_matrix(log_path, selectivity_threshold = 0.1):

    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    features = [d['feature'] for d in data]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))

    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)
    for d in data:
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), d['feature']] = importance

    # We normalize by row to see the distribution of the atom's importance across features
    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_relative = matrix.divide(row_sums, axis=0)

    # If the atom assigns less than X% of its importance to a feature, it is deleted
    matrix_clean = matrix.where(matrix_relative > selectivity_threshold, 0)

    # 3. Sorting
    max_feature_idx = matrix_clean.idxmax(axis=1).map({feat: i for i, feat in enumerate(features)})
    max_values = matrix_clean.max(axis=1)

    sort_df = pd.DataFrame({
        'feature_pos': max_feature_idx,
        'importance': max_values
    }, index=matrix_clean.index)

    sorted_atoms = sort_df.sort_values(by=['feature_pos', 'importance'], ascending=[True, False]).index

    matrix_final = matrix_clean.loc[sorted_atoms]
    #matrix_final = matrix_final[matrix_final.sum(axis=1) > 0]

    plt.figure(figsize=(28, 15))

    ax = sns.heatmap(matrix_final.T, cmap="YlGnBu", cbar_kws={'label': 'Importance'}, xticklabels=True, yticklabels=True)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=15)
    plt.xlabel("Atoms", fontsize=19, labelpad=15)
    plt.ylabel("Features", fontsize=19, labelpad=15)
    plt.tight_layout()
    plt.show()

