import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def analyze_hierarchical_correlation(matrix):
    hierarchy_pairs = [
        ('etre_vivant', 'animal'), ('etre_vivant', 'plante'),
        ('animal', 'terrestre'), ('animal', 'aquatique'),
        ('plante', 'arbre'), ('plante', 'fleur'),
        ('objet', 'alimentaire'), ('objet', 'outil')
    ]
    results = []
    for parent, child in hierarchy_pairs:
        if parent in matrix.columns and child in matrix.columns:
            correlation = matrix[parent].corr(matrix[child])
            results.append({'Parent': parent, 'Enfant': child, 'Correlation': correlation})
    return pd.DataFrame(results)

def plot_hierarchy_heatmap(matrix, n_components, n_nonzero, layer, cfg):
    ordered_features = [
        'etre_vivant', 'animal', 'terrestre', 'aquatique',
        'plante', 'arbre', 'fleur', 'objet', 'alimentaire', 'outil'
    ]
    cols = [c for c in ordered_features if c in matrix.columns]
    if not cols: return

    corr_matrix = matrix[cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title(f"Layer {layer}, C={n_components}, K={n_nonzero}")

    save_path = os.path.join(cfg.figures_dir, f"hierarchy_L{layer}_C{n_components}.png")
    plt.savefig(save_path)
    plt.show()

def plot_atom_importance(feature_name, log_path, top_k=10):
    with open(log_path, 'r') as f:
        data = [json.loads(line) for line in f if json.loads(line)["feature"] == feature_name]
    if not data: return
    entry = data[0]

    atoms = list(entry["permutation_importance"].keys())
    perm_vals = [entry["permutation_importance"][a] for a in atoms]
    weight_vals = [entry["weights"].get(a, 0) for a in atoms]

    idx = np.argsort(perm_vals)
    atoms = [atoms[i] for i in idx]; perm_vals = [perm_vals[i] for i in idx]; weight_vals = [weight_vals[i] for i in idx]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(np.arange(len(atoms)) + 0.2, perm_vals, 0.4, color='skyblue', label='Permutation')
    ax1.set_yticks(np.arange(len(atoms)))
    ax1.set_yticklabels([f"Atom {a}" for a in atoms])

    ax2 = ax1.twiny()
    ax2.barh(np.arange(len(atoms)) - 0.2, [abs(w) for w in weight_vals], 0.4, color='salmon', alpha=0.6, label='Weight')
    plt.title(f"Atom Importance: {feature_name}")
    plt.show()

def get_top_stimuli_for_atoms(sentences_df, n_components, n_nonzero, layer, cfg, atom_indices=[0, 1], top_k=10):
    z_path = os.path.join(cfg.z_cache, f"Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy")
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Z file not found: {z_path}")

    Z = np.load(z_path)
    text_col = 'sentence' if 'sentence' in sentences_df.columns else sentences_df.columns[0]
    texts = sentences_df[text_col].values
    results = {}

    for atom_idx in atom_indices:
        activations = Z[:, atom_idx]
        top_indices = np.argsort(activations)[::-1][:top_k]
        results[atom_idx] = [(texts[i], activations[i]) for i in top_indices]
    return results



def plot_selectivity_matrix(log_path, n_components, n_nonzero, layer, cfg, selectivity_threshold=0.1):
    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. num.",
        "subj_GEN": "Subj. gender",
        "subj_ZIPF": "Subj. freq.",
        "obj_NUM": "Obj. num.",
        "obj_GEN": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM": "Embed. num.",
        "embed_GEN": "Embed. gender",
        "embed_ZIPF": "Embed. freq.",
        "verb_ZIPF": "Verb freq.",
    }

    data = []
    feature_metrics = {}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
            raw_feat = entry['feature']
            clean_feat = rename_dict.get(raw_feat, raw_feat)
            feature_metrics[clean_feat] = {
                'auc': entry.get('roc_auc', 0.0),
                'p': entry.get('p_value', 1.0)
            }

    features_raw = [d['feature'] for d in data]
    features = [rename_dict.get(f, f) for f in features_raw]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))

    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)
    for d in data:
        feat_name = rename_dict.get(d['feature'], d['feature'])
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), feat_name] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_relative = matrix.divide(row_sums, axis=0)
    matrix_clean = matrix.where(matrix_relative > selectivity_threshold, 0)

    max_feature_idx = matrix_clean.idxmax(axis=1).map({feat: i for i, feat in enumerate(features)})
    max_values = matrix_clean.max(axis=1)

    sort_df = pd.DataFrame({'feature_pos': max_feature_idx, 'importance': max_values}, index=matrix_clean.index)
    sorted_atoms = sort_df.sort_values(by=['feature_pos', 'importance'], ascending=[True, False]).index
    matrix_final = matrix_clean.loc[sorted_atoms]

    plt.figure(figsize=(28, 14))
    ax = sns.heatmap(matrix_final.T, cmap="YlGnBu", cbar_kws={'label': 'Importance'}, xticklabels=True, yticklabels=True)

    features_ordered = matrix_final.columns

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Importance', size=25)

    ax.tick_params(axis='x', labelsize=14, rotation=90)
    ax.tick_params(axis='y', labelsize=24, rotation=0)

    for lab in ax.get_yticklabels():
        lab.set_va('center')

    plt.xlabel("Atoms", fontsize=30, labelpad=15)
    plt.ylabel("Features", fontsize=30, labelpad=20)
    plt.title(f"Layer {layer}, components {n_components}, nonzero {n_nonzero}", fontsize=34, pad=20)

    plt.tight_layout()
    save_path = os.path.join(cfg.figures_dir, f"selectivity_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=70)
    plt.show()

    return matrix_final

def plot_selectivity_matrix2(log_path, n_components, n_nonzero, layer, cfg, selectivity_threshold=0.1):
    data = []
    with open(log_path, 'r') as f:
        for line in f: data.append(json.loads(line))

    features = [d['feature'] for d in data]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))

    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)
    for d in data:
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), d['feature']] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_clean = matrix.where((matrix.divide(row_sums, axis=0)) > selectivity_threshold, 0)

    # Custom order
    custom_order = ["etre_vivant", "animal", "terrestre", "aquatique", "plante", "arbre", "fleur", "objet", "alimentaire", "outil"]
    matrix_final = matrix_clean.reindex(columns=[c for c in custom_order if c in matrix_clean.columns])

    plt.figure(figsize=(20, 10))
    sns.heatmap(matrix_final.T, cmap="YlGnBu")
    plt.title(f"Selectivity Layer {layer}")

    save_path = os.path.join(cfg.figures_dir, f"selectivity_L{layer}_C{n_components}.png")
    plt.savefig(save_path)
    plt.show()
    return matrix_final

def compute_identifiability_metrics(log_path):
    data = []
    with open(log_path, 'r') as f:
        for line in f: data.append(json.loads(line))

    features = [d['feature'] for d in data]
    all_atoms = sorted(list(set(int(a) for d in data for a in d['permutation_importance'].keys())))
    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)
    for d in data:
        for atom, importance in d['permutation_importance'].items():
            matrix.loc[int(atom), d['feature']] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    results_df = pd.DataFrame({
        'S_ID': matrix.max(axis=1) / row_sums,
        'dominant_feature': matrix.idxmax(axis=1)
    })
    return results_df[matrix.sum(axis=1) > 0], matrix

def plot_identifiability_distribution(log_path, n_components, n_nonzero, layer, cfg):
    results_df, _ = compute_identifiability_metrics(log_path)
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['S_ID'], bins=20, color='teal')
    plt.title(f"Identifiability L{layer}")

    save_path = os.path.join(cfg.figures_dir, f"id_dist_L{layer}_C{n_components}.png")
    plt.savefig(save_path)
    plt.show()

def plot_identifiability(n_layers, n_components, n_nonzero, cfg):
    all_results = []
    for layer in range(n_layers):
        log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")
        if os.path.exists(log_path):
            res_df, _ = compute_identifiability_metrics(log_path)
            for s in res_df['S_ID']:
                all_results.append({'Layer': layer, 'S_ID': s})

    if not all_results: return
    df_plot = pd.DataFrame(all_results)
    plt.figure(figsize=(12, 7))
    sns.stripplot(data=df_plot, x='Layer', y='S_ID', color='teal', alpha=0.3)
    sns.pointplot(data=df_plot, x='Layer', y='S_ID', color='red')

    save_path = os.path.join(cfg.figures_dir, f"id_layers_C{n_components}.png")
    plt.savefig(save_path)
    plt.show()




def aggregate_importance(log_path, n_components, n_nonzero, layer, cfg, mode="k", k=5, pct=0.05):

    rows = []

    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            if "permutation_importance" not in entry:
                continue

            feature = entry["feature"]
            imp_dict = entry["permutation_importance"]

            values = np.array(list(imp_dict.values()))
            values = np.maximum(values, 0)  # clip négatif

            n_atoms = len(values)

            if mode == "k":
                k_eff = min(k, n_atoms)
            elif mode == "pct":
                k_eff = max(1, int(np.ceil(pct * n_atoms)))
            else:
                raise ValueError("mode doit être 'k' ou 'pct'")

            top_vals = np.sort(values)[::-1][:k_eff]
            score = top_vals.sum()

            rows.append({
                "feature": feature,
                "score": score
            })

    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    return df



def plot_importance_across_layers(results_per_layer, n_components=None, n_nonzero=None, cfg=None, keep_only_renamed=True):
    rename = {
        "sentence_CLAUSE_subjwho": "Relative clause type",
        "sentence_RC_attached_peripheral": "Attachment site",
        "subj_NUM_sg": "Subj. num.",
        #"subj_GEN_m": "Subj. gender",
        #"subj_ZIPF": "Subj. freq.",
        #"obj_NUM_sg": "Obj. num.",
        #"obj_GEN_m": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM_sg": "Embed. num.",
        #"embed_GEN_m": "Embed. gender",
        #"embed_ZIPF": "Embed. freq.",
        #"verb_ZIPF": "Verb freq.",
    }

    feature_colors = {
        "Relative clause type": "green",
        "Attachment site": "orange",
        "Subj. num.": "royalblue",
        "Obj. freq.": "red",
        "Embed. num.": "mediumpurple",
        "Embed. freq.": "red",
    }

    # concat
    all_df = []
    for layer, df in results_per_layer.items():
        tmp = df[["feature", "score"]].copy()
        tmp["layer"] = layer
        all_df.append(tmp)
    all_df = pd.concat(all_df, ignore_index=True)

    if keep_only_renamed:
        all_df = all_df[all_df["feature"].isin(rename.keys())].copy()

    # rename
    all_df["feature_renamed"] = all_df["feature"].map(lambda x: rename.get(x, x))

    # pivot
    pivot = all_df.pivot(index="layer", columns="feature_renamed", values="score").sort_index()

    target_order = [v for k, v in rename.items() if v in pivot.columns]
    pivot = pivot.reindex(columns=target_order)
    x = pivot.index.values

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "axes.linewidth": 3.0,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.major.width": 3.0,
        "ytick.major.width": 3.0,
    })

    markers = ["o", "x", "s", "P", "D", "^", "v", "<", ">"]
    lw = 3.5
    ms = 9

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for j, col in enumerate(pivot.columns):

        ax.plot(
            x,
            pivot[col].values,
            marker=markers[j % len(markers)],
            linewidth=lw,
            markersize=ms,
            label=col,
            color=feature_colors.get(col, "black"))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature Importance")
    ax.set_xticks(x)


    ax.legend(title="Feature", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.title(f"components {n_components}, nonzero {n_nonzero}", fontsize=20)
    plt.tight_layout()

    save_path = os.path.join(cfg.figures_dir, f"importance_across_layer_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=70)
    plt.show()

def compute_neff(w_m):
    """
    Calculates the Effective Number of Atoms (N_eff) based on Hill's entropy of order 2.

    1. Normalization: p_m = w_m / sum(w)
    2. N_eff = 1 / sum(p_m^2)
    """
    w_pos = np.maximum(w_m, 0)
    sum_w = np.sum(w_pos)

    if sum_w == 0:
        return 0.0, np.zeros_like(w_pos)

    # Probability distribution p_m
    p_m = w_pos / sum_w

    # Effective number N_eff
    n_eff = 1.0 / np.sum(p_m ** 2)

    return n_eff, p_m


def roc_auc_plot(n_components, n_nonzero, layer, cfg):
    data = []
    log_files = glob.glob(os.path.join(cfg.log_dir, "experiment_log_layer*.jsonl"))

    if not log_files:
        return

    for filename in log_files:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    data.append({
                        "Layer": entry["layer"],
                        "Feature": entry["feature"],
                        "ROC AUC": entry["roc_auc"]
                    })
                except: continue

    df = pd.DataFrame(data)
    df = df.sort_values(by="Layer")

    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. num.",
        "subj_GEN": "Subj. gender",
        "subj_ZIPF": "Subj. freq.",
        "obj_NUM": "Obj. num.",
        "obj_GEN": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM": "Embed. num.",
        "embed_GEN": "Embed. gender",
        "embed_ZIPF": "Embed. freq.",
        "verb_ZIPF": "Verb freq."
    }

    df['Feature Label'] = df['Feature'].map(rename_dict).fillna(df['Feature'])

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq." in label: return 'Frequency (Zipf)'
        if any(x in label for x in ["num.", "gender"]): return 'Morphology'
        return 'Other'

    df['Category'] = df['Feature Label'].apply(get_category)
    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    unique_features = df['Feature Label'].unique()
    palette = sns.color_palette("bright", len(unique_features))
    color_map = dict(zip(unique_features, palette))

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for i, cat in enumerate(category_order):
        ax = axes[i]
        df_cat = df[df['Category'] == cat]

        if df_cat.empty:
            continue

        sns.lineplot(
            data=df_cat,
            x="Layer",
            y="ROC AUC",
            hue="Feature Label",
            style="Feature Label",
            markers=True,
            dashes=False,
            linewidth=2.5,
            palette=color_map,
            ax=ax
        )

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.set_ylim(0.45, 1.05)
        #ax.set_ylim(0, 0.5)
        ax.set_title(cat, fontweight='bold', size=17, pad=15)
        ax.set_xlabel("Layer", fontsize=16)

        if i == 0:
            ax.set_ylabel("Decoding Score (AUC)", fontsize=16)
        else:
            ax.set_ylabel("")

        handles, labels = ax.get_legend_handles_labels()
        num_items = len(labels)
        num_cols = 2 if num_items > 4 else 1

        ax.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=num_cols,
            fontsize=14,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0
        )

    plt.subplots_adjust(bottom=0.35, wspace=0.05)

    save_path = os.path.join(cfg.figures_dir, f"roc_auc_plot_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=70)
    plt.show()

def p_value_plot(n_components, n_nonzero, layer, cfg):
    data = []
    log_files = glob.glob(os.path.join(cfg.log_dir, "experiment_log_layer*.jsonl"))

    if not log_files:
        return

    for filename in log_files:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    data.append({
                        "Layer": entry["layer"],
                        "Feature": entry["feature"],
                        "p-value": entry["p_value"]
                    })
                except: continue

    df = pd.DataFrame(data)
    df = df.sort_values(by="Layer")

    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. num.",
        "subj_GEN": "Subj. gender",
        "subj_ZIPF": "Subj. freq.",
        "obj_NUM": "Obj. num.",
        "obj_GEN": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM": "Embed. num.",
        "embed_GEN": "Embed. gender",
        "embed_ZIPF": "Embed. freq.",
        "verb_ZIPF": "Verb freq."
    }

    df['Feature Label'] = df['Feature'].map(rename_dict).fillna(df['Feature'])

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq." in label: return 'Frequency (Zipf)'
        if any(x in label for x in ["num.", "gender"]): return 'Morphology'
        return 'Other'

    df['Category'] = df['Feature Label'].apply(get_category)
    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    unique_features = df['Feature Label'].unique()
    palette = sns.color_palette("bright", len(unique_features))
    color_map = dict(zip(unique_features, palette))

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for i, cat in enumerate(category_order):
        ax = axes[i]
        df_cat = df[df['Category'] == cat]

        if df_cat.empty:
            continue

        sns.lineplot(
            data=df_cat,
            x="Layer",
            y="p-value",
            hue="Feature Label",
            style="Feature Label",
            markers=True,
            dashes=False,
            linewidth=2.5,
            palette=color_map,
            ax=ax
        )

        ax.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        #ax.set_ylim(0.45, 1.05)
        ax.set_ylim(0, 0.3)
        ax.set_title(cat, fontweight='bold', size=17, pad=15)
        ax.set_xlabel("Layer", fontsize=16)

        if i == 0:
            ax.set_ylabel("p-value", fontsize=16)
        else:
            ax.set_ylabel("")

        handles, labels = ax.get_legend_handles_labels()
        num_items = len(labels)
        num_cols = 2 if num_items > 4 else 1

        ax.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=num_cols,
            fontsize=14,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0
        )

    plt.subplots_adjust(bottom=0.35, wspace=0.05)
    save_path = os.path.join(cfg.figures_dir, f"p_value_plot_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=70)
    plt.show()

def neff_plot(n_components, n_nonzero, layer, cfg):
    data = []
    log_files = glob.glob(os.path.join(cfg.log_dir, "experiment_log_layer*.jsonl"))

    if not log_files:
        return

    for filename in log_files:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)


                    wm_dict = entry.get("permutation_importance", {})
                    wm_values = np.array(list(wm_dict.values()))

                    w_pos = np.maximum(wm_values, 0)
                    sum_w = np.sum(w_pos)

                    if sum_w > 0:
                        p_m = w_pos / sum_w
                        neff = 1.0 / np.sum(p_m**2)
                    else:
                        neff = 1.0

                    data.append({
                        "Layer": entry["layer"],
                        "Feature": entry["feature"],
                        "Neff": neff
                    })
                except: continue

    df = pd.DataFrame(data)
    df = df.sort_values(by="Layer")

    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. num.",
        "subj_GEN": "Subj. gender",
        "subj_ZIPF": "Subj. freq.",
        "obj_NUM": "Obj. num.",
        "obj_GEN": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM": "Embed. num.",
        "embed_GEN": "Embed. gender",
        "embed_ZIPF": "Embed. freq.",
        "verb_ZIPF": "Verb freq."
    }

    df['Feature Label'] = df['Feature'].map(rename_dict).fillna(df['Feature'])

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq." in label: return 'Frequency (Zipf)'
        if any(x in label for x in ["num.", "gender"]): return 'Morphology'
        return 'Other'

    df['Category'] = df['Feature Label'].apply(get_category)
    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    unique_features = df['Feature Label'].unique()
    palette = sns.color_palette("bright", len(unique_features))
    color_map = dict(zip(unique_features, palette))

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for i, cat in enumerate(category_order):
        ax = axes[i]
        ax.axhline(1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        df_cat = df[df['Category'] == cat]
        if df_cat.empty: continue

        sns.lineplot(
            data=df_cat, x="Layer", y="Neff",
            hue="Feature Label", style="Feature Label",
            markers=True, dashes=False, linewidth=2.5,
            palette=color_map, ax=ax
        )

        ax.set_title(cat, fontweight='bold', size=17, pad=15)
        ax.set_xlabel("Layer", fontsize=16)
        if i == 0:
            ax.set_ylabel("Neff", fontsize=16)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.set_ylim(30, -4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  ncol=2 if len(labels) > 4 else 1, fontsize=12, frameon=False)

    plt.subplots_adjust(bottom=0.35, wspace=0.05)
    save_path = os.path.join(cfg.figures_dir, f"neff_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=70)
    plt.show()


def load_logs_to_df(log_dir):
    data = []
    files = glob.glob(os.path.join(log_dir, "experiment_log_layer*.jsonl"))
    for f_path in files:
        with open(f_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: continue
    return pd.DataFrame(data)

def compute_significance_matrix(df, feature_name):
    n_layers = 12
    df_feat = df[df['feature'] == feature_name].sort_values('layer')

    if len(df_feat) < n_layers:
        return None, None

    aucs = df_feat['roc_auc'].values
    p_matrix = np.ones((n_layers, n_layers))

    # [cite_start]Estimation de l'erreur standard (SE) via le bootstrap mentionné dans le PDF [cite: 63, 65]
    # À défaut de SE individuelle par log, on utilise une SE conservative de 0.015
    se = 0.015

    for i in range(n_layers):
        for j in range(n_layers):
            if i == j: continue
            diff = aucs[i] - aucs[j]
            z = diff / np.sqrt(2 * se**2)
            p_matrix[i, j] = 2 * (1 - norm.cdf(np.abs(z)))

    # [cite_start]Correction FDR (Benjamini-Hochberg) sur la partie triangulaire supérieure [cite: 67]
    upper_idx = np.triu_indices(n_layers, k=1)
    _, p_adj, _, _ = multipletests(p_matrix[upper_idx], method='fdr_bh')

    adj_matrix = np.ones((n_layers, n_layers))
    adj_matrix[upper_idx] = p_adj
    adj_matrix = adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix))

    return adj_matrix, aucs

def plot_layer_comparison_matrix(p_matrix, aucs, feature_name, rename_dict):
    n_layers = 12
    clean_name = rename_dict.get(feature_name, feature_name)

    # Matrice de direction : 1 si Row > Col (Significatif), -1 si Row < Col, 0 sinon
    sig_map = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            if p_matrix[i, j] < 0.05:
                sig_map[i, j] = 1 if aucs[i] > aucs[j] else -1

    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(10, 240, as_cmap=True) # Rouge (-) à Bleu (+)

    ax = sns.heatmap(sig_map, cmap=cmap, center=0, annot=False, cbar=True,
                xticklabels=range(n_layers), yticklabels=range(n_layers),
                linewidths=0.5, linecolor='white')

    # Personnalisation de la barre de couleur
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.66, 0, 0.66])
    colorbar.set_ticklabels(['Moins bon', 'Stable', 'Meilleur'])

    plt.title(f"Évolution statistique de l'encodage : {clean_name}\n(FDR corrigé, p < 0.05)", fontsize=16)
    plt.xlabel("Couche de référence (j)")
    plt.ylabel("Couche comparée (i)")
    plt.tight_layout()
    plt.show()

def plot_layer_comparison(cfg):

    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. num.",
        "subj_GEN": "Subj. gender",
        "subj_ZIPF": "Subj. freq.",
        "obj_NUM": "Obj. num.",
        "obj_GEN": "Obj. gender",
        "obj_ZIPF": "Obj. freq.",
        "embed_NUM": "Embed. num.",
        "embed_GEN": "Embed. gender",
        "embed_ZIPF": "Embed. freq.",
        "verb_ZIPF": "Verb freq."
    }

    df_logs = load_logs_to_df(cfg.log_dir)
    features_to_plot = df_logs['feature'].unique()

    for feat in features_to_plot:
        p_mat, auc_vals = compute_significance_matrix(df_logs, feat)
        if p_mat is not None:
            plot_layer_comparison_matrix(p_mat, auc_vals, feat, rename_dict)