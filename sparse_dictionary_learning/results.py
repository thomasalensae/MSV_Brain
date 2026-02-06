import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_hierarchical_correlation(matrix):
    hierarchy_pairs = [
        ('etre_vivant', 'animal'),
        ('etre_vivant', 'plante'),
        ('animal', 'terrestre'),
        ('animal', 'aquatique'),
        ('plante', 'arbre'),
        ('plante', 'fleur'),
        ('objet', 'alimentaire'),
        ('objet', 'outil')
    ]

    results = []
    for parent, child in hierarchy_pairs:
        if parent in matrix.columns and child in matrix.columns:

            correlation = matrix[parent].corr(matrix[child])
            results.append({
                'Parent': parent,
                'Enfant': child,
                'Correlation': correlation
            })

    return pd.DataFrame(results)

def plot_hierarchy_heatmap(matrix, n_components=512, n_nonzero=20, layer=6):
    ordered_features = [
        'etre_vivant', 'animal', 'terrestre', 'aquatique',
        'plante', 'arbre', 'fleur',
        'objet', 'alimentaire', 'outil'
    ]

    cols = [c for c in ordered_features if c in matrix.columns]
    corr_matrix = matrix[cols].corr()

    plt.figure(figsize=(12, 10))

    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", annot_kws={"size": 12, "color": "black"}, cmap='RdBu_r', center=0, vmin=-1, vmax=1)

    plt.xticks(fontsize=18, rotation=45, ha='right')
    plt.yticks(fontsize=18, rotation=0)

    plt.title(f"Layer {layer}, components {n_components}, nonzero {n_nonzero}", fontsize=20)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(f"/Users/maxime/MSV_Brain/sparse_dictionary_learning/figures/correlation_hierarchique_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}")
    plt.show()


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

    Z_path=f"sparse_dictionary_learning/cache/Z_cache/Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"
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

def plot_selectivity_matrix(log_path, n_components, n_nonzero, layer, selectivity_threshold = 0.1, ):

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

    order = True
    if order:
        custom_feature_order = [
            "animal",
            "animal_terrestre",
            "animal_aquatique",
            "plante",
            "plante_arbre",
            "plante_fleur",
            "objet",
            "objet_alimentaire",
            "objet_non_alimentaire"
        ]
        custom_feature_order = [
            "etre_vivant",
            "animal",
            "terrestre",
            "aquatique",
            "plante",
            "arbre",
            "fleur",
            "objet",
            "alimentaire",
            "outil"
        ]

        matrix_final = matrix_final.reindex(columns=custom_feature_order)

    plt.figure(figsize=(28, 15))

    ax = sns.heatmap(matrix_final.T, cmap="YlGnBu", cbar_kws={'label': 'Importance'}, xticklabels=True, yticklabels=True)
    ax.figure.axes[-1].yaxis.label.set_size(25)
    ax.figure.axes[-1].tick_params(labelsize=20)
    ax.tick_params(axis='x', labelsize=18, rotation=70)
    ax.tick_params(axis='y', labelsize=28, rotation=45)
    for lab in ax.get_yticklabels():
        lab.set_va('top')
    plt.xlabel("Atoms", fontsize=27, labelpad=15)
    plt.ylabel("Features", fontsize=27, labelpad=15)
    plt.title(f"Layer {layer}, components {n_components}, nonzero {n_nonzero}", fontsize=34)
    plt.tight_layout()
    save_path = f"sparse_dictionary_learning/figures/selectivity_matrix_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    print(f"Figure saved to {save_path}")
    plt.show()

    return matrix_final



def compute_identifiability_metrics(log_path, threshold=0.8):
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Matrix Atoms x Features with the importance of permutation
    features = [d['feature'] for d in data]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))
    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)

    for d in data:
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), d['feature']] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    id_scores = matrix.max(axis=1) / row_sums # importance of the atom on its prefered feature / importance total on all features

    # Find the prefered feature for each atom
    dominant_feature = matrix.idxmax(axis=1)

    results_df = pd.DataFrame({
        'S_ID': id_scores,
        'dominant_feature': dominant_feature
    })

    # Remove the atoms with null importance
    results_df = results_df[matrix.sum(axis=1) > 0]

    return results_df, matrix
def plot_identifiability_distribution(log_path, n_components, n_nonzero, layer):

    results_df, matrix = compute_identifiability_metrics(log_path=log_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['S_ID'], bins=20, kde=True, color='teal')
    plt.title(f"Layer {layer}, components {n_components}, nonzero {n_nonzero}",fontsize=16)
    plt.xlabel("Identifiability Score",fontsize=16)
    plt.ylabel("Number of Atoms",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.legend()

    save_path = f"sparse_dictionary_learning/figures/identifiability_distribution_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

    plt.show()


def compute_sid_for_layer(log_path):
    if not os.path.exists(log_path):
        return None

    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        return None

    features = [d['feature'] for d in data]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))
    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)

    for d in data:
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), d['feature']] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    all_sid_scores = matrix.max(axis=1) / row_sums

    top_atoms_indices = matrix.idxmax(axis=0)

    top_sid_scores = [all_sid_scores.loc[atom_idx] for atom_idx in top_atoms_indices]

    return top_sid_scores

def plot_identifiability(n_layers, n_components, n_nonzero):

    all_results = []

    for layer in range(0, n_layers):

        log_path = f"sparse_dictionary_learning/cache/log/experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl"
        scores = compute_sid_for_layer(log_path)

        if scores is not None:
            for s in scores:
                all_results.append({'Layer': layer, 'S_ID': s})

    df_plot = pd.DataFrame(all_results)

    plt.figure(figsize=(12, 7))

    sns.stripplot(data=df_plot, x='Layer', y='S_ID', size=5, color='teal', alpha=0.5, jitter=0.25)
    sns.pointplot(data=df_plot, x='Layer', y='S_ID', color='red', scale=0.7, label='Average')

    plt.title(f"components {n_components}, nonzero {n_nonzero}", fontsize=16)
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Identifiability Score", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    save_path = f"sparse_dictionary_learning/figures/identifiability_per_layer_figures_ncomp{n_components}_nnonzero{n_nonzero}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()
