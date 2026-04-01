import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MaxNLocator

dpi = 70

all_features_ordered = ['Verb freq.', 'Relative clause type', 'Attachment site', 'Subj. num.',
 'Embed. freq.', 'Obj. freq.', 'Subj. freq.', 'Embed. gender', 'Embed. num.',
 'Obj. gender', 'Obj. num.', 'Subj. gender']


global_colors = sns.color_palette("tab10", len(all_features_ordered))
color_map = dict(zip(all_features_ordered, global_colors))

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

def plot_matrix(log_path, n_components, n_nonzero, layer, cfg, selectivity_threshold=0.2):

    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    features = list(rename_dict.values())
    all_atoms = sorted(list(set(int(a) for d in data for a in d.get("permutation_importance", {}).keys())))

    matrix_imp = pd.DataFrame(0.0, index=all_atoms, columns=features)
    matrix_coef = pd.DataFrame(0.0, index=all_atoms, columns=features)

    for d in data:
        if d['feature'] in rename_dict:
            feat_name = rename_dict[d['feature']]
            for atom, importance in d.get("permutation_importance", {}).items():
                matrix_imp.loc[int(atom), feat_name] = importance
            for atom, coef_val in d.get("coef", {}).items():
                matrix_coef.loc[int(atom), feat_name] = coef_val

    row_sums = matrix_imp.sum(axis=1).replace(0, 1)
    matrix_relative = matrix_imp.divide(row_sums, axis=0)
    matrix_clean = matrix_imp.where(matrix_relative > selectivity_threshold, 0)

    max_values_per_atom = matrix_clean.max(axis=1)
    atoms_to_keep = max_values_per_atom[max_values_per_atom > 0].index


    matrix_clean = matrix_clean.loc[atoms_to_keep]
    matrix_coef = matrix_coef.loc[atoms_to_keep]

    max_feature_idx = matrix_clean.idxmax(axis=1).map({feat: i for i, feat in enumerate(features)})
    sort_df = pd.DataFrame({'feature_pos': max_feature_idx, 'importance': matrix_clean.max(axis=1)}, index=matrix_clean.index)
    sorted_atoms = sort_df.sort_values(by=['feature_pos', 'importance'], ascending=[True, False]).index

    matrix_imp_final = matrix_clean.loc[sorted_atoms]
    matrix_coef_final = np.abs(matrix_coef.loc[sorted_atoms])

    N_feat = len(matrix_imp_final.columns)
    N_atoms = len(matrix_imp_final.index)

    dynamic_width = max(15, N_atoms * 0.6)

    grid_imp = np.full((2 * N_feat, N_atoms), np.nan)
    grid_coef = np.full((2 * N_feat, N_atoms), np.nan)
    grid_imp[0::2, :] = matrix_imp_final.T.values
    grid_coef[1::2, :] = matrix_coef_final.T.values

    fig, ax = plt.subplots(figsize=(dynamic_width, 10))

    cmap_imp = plt.get_cmap("YlGnBu").copy()
    cmap_imp.set_bad(color='none')
    im_imp = ax.imshow(grid_imp, cmap=cmap_imp, aspect="auto", interpolation="none")

    cmap_coef = plt.get_cmap("Reds").copy()
    cmap_coef.set_bad(color='none')
    im_coef = ax.imshow(grid_coef, cmap=cmap_coef, aspect="auto", interpolation="none")

    for y in range(0, 2 * N_feat + 1, 2):
        ax.axhline(y - 0.5, color='black', linewidth=0.5)

    ax.set_xticks(np.arange(N_atoms))
    ax.set_xticklabels(matrix_imp_final.index, rotation=90, fontsize=14)
    ax.set_xlabel("Atoms", fontsize=30, labelpad=15, fontweight='bold')

    ax.set_yticks(np.arange(0, 2 * N_feat, 2) + 0.5)
    ax.set_yticklabels(matrix_imp_final.columns, fontsize=28)
    ax.set_ylabel("Features", fontsize=30, labelpad=20, fontweight='bold')

    ax.tick_params(axis='both', which='both', length=0)
    plt.title(f"Layer {layer}", fontsize=34, pad=20, fontweight='bold')

    plt.subplots_adjust(left=0.12, right=0.82, top=0.90, bottom=0.15)

    pos = ax.get_position()

    cax_imp = fig.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.55, 0.012, pos.height * 0.40])
    cbar_imp = plt.colorbar(im_imp, cax=cax_imp)
    cbar_imp.ax.tick_params(labelsize=18)
    cbar_imp.set_label('Importance', size=24, fontweight='bold', labelpad=15)

    cax_coef = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.012, pos.height * 0.40])
    cbar_coef = plt.colorbar(im_coef, cax=cax_coef)
    cbar_coef.ax.tick_params(labelsize=18)
    cbar_coef.set_label('|coef|', size=24, fontweight='bold', labelpad=15)

    save_path = os.path.join(cfg.figures_dir, f"importance_split_matrix_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_importance_matrix(log_path, n_components, n_nonzero, layer, cfg, selectivity_threshold=0.1):

    data = []
    feature_metrics = {}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
            raw_feat = entry['feature']
            if raw_feat in rename_dict:
                clean_feat = rename_dict[raw_feat]
                feature_metrics[clean_feat] = {
                    'auc': entry.get('roc_auc', 0.0),
                    'p': entry.get('p_value', 1.0)
                }

    features = list(rename_dict.values())
    all_atoms = sorted(list(set(int(a) for d in data for a in d.get("permutation_importance", {}).keys())))

    matrix = pd.DataFrame(0.0, index=all_atoms, columns=features)
    for d in data:
        if d['feature'] in rename_dict:
            feat_name = rename_dict[d['feature']]
            for atom, importance in d.get("permutation_importance", {}).items():
                matrix.loc[int(atom), feat_name] = importance

    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_relative = matrix.divide(row_sums, axis=0)
    matrix_clean = matrix.where(matrix_relative > selectivity_threshold, 0)

    max_feature_idx = matrix_clean.idxmax(axis=1).map({feat: i for i, feat in enumerate(features)})
    max_values = matrix_clean.max(axis=1)

    sort_df = pd.DataFrame({'feature_pos': max_feature_idx, 'importance': max_values}, index=matrix_clean.index)
    sorted_atoms = sort_df.sort_values(by=['feature_pos', 'importance'], ascending=[True, False]).index
    matrix_final = matrix_clean.loc[sorted_atoms]

    plt.figure(figsize=(28, 10))
    ax = sns.heatmap(
        matrix_final.T,
        cmap="YlGnBu",
        cbar_kws={'label': 'Importance', 'pad': 0.02, 'shrink': 0.8},
        xticklabels=True,
        yticklabels=True
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Importance', size=30, fontweight='bold', labelpad=20)

    ax.tick_params(axis='x', labelsize=14, rotation=90)
    ax.tick_params(axis='y', labelsize=28, rotation=0)

    for lab in ax.get_yticklabels():
        lab.set_va('center')

    plt.xlabel("Atoms", fontsize=30, labelpad=15, fontweight='bold')
    plt.ylabel("Features", fontsize=30, labelpad=20, fontweight='bold')
    plt.title(f"Layer {layer}", fontsize=34, pad=20, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(cfg.figures_dir, f"importance_matrix_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

    return matrix_final

def plot_conditional_heatmap(X, layer, n_components, n_nonzero, feature_name, cfg, top_k=15):

    z_file_name = f"Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"
    z_path = os.path.join(cfg.z_cache, z_file_name)

    if not os.path.exists(z_path):
        print(f"Erreur : Matrice {z_file_name} introuvable.")
        return

    Z = np.load(z_path)

    log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")

    importances = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['feature'] == feature_name:
                    importances = entry.get('permutation_importance', {})
                    break


    sorted_atoms = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    top_atoms_indices = [int(atom) for atom, score in sorted_atoms[:top_k] if score > 0]

    if len(top_atoms_indices) < 2:
        return

    condition_mask = X[feature_name] == 1
    if not condition_mask.any():
        return

    Z_filtered = Z[condition_mask]

    Z_top = Z_filtered[:, top_atoms_indices]
    labels = [f"Atom {i}" for i in top_atoms_indices]
    df_Z_top = pd.DataFrame(Z_top, columns=labels)

    corr_matrix = df_Z_top.corr()

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        corr_matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .8}
    )

    plt.title(
        f"Pearson correlation : {feature_name}\n",
        fontsize=16, fontweight='bold', pad=20
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_path = os.path.join(cfg.figures_dir, f"heatmap_corr_{feature_name}_L{layer}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


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


    df['Feature Label'] = df['Feature'].map(rename_dict).fillna(df['Feature'])

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq." in label: return 'Frequency (Zipf)'
        if any(x in label for x in ["num.", "gender"]): return 'Morphology'
        return 'Other'

    df['Category'] = df['Feature Label'].apply(get_category)
    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

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

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.2)
        ax.set_ylim(0.45, 1.05)
        ax.set_title(cat, fontweight='bold', size=18, pad=15)
        ax.set_xlabel("Layer", fontsize=18)
        ax.tick_params(axis='both', labelsize=14)

        if i == 0:
            ax.set_ylabel("Decoding Score (AUC)", fontsize=18)
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
            fontsize=15,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0
        )

    plt.subplots_adjust(bottom=0.35, wspace=0.05)

    save_path = os.path.join(cfg.figures_dir, f"roc_auc_plot_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=dpi)
    plt.show()


def neff_plot_cv(n_components, n_nonzero, cfg):
    data = []

    pattern = os.path.join(cfg.log_dir_cv, f"experiment_log_layer*_ncomp{n_components}_nnonzero{n_nonzero}_fold*.jsonl")
    log_files = glob.glob(pattern)

    if not log_files:
        print(f"No file found in {cfg.log_dir_cv} with C={n_components}, K={n_nonzero}")
        return

    for filename in log_files:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    wm_dict = entry.get("permutation_importance", {})
                    wm_values = np.array(list(wm_dict.values()))

                    # Calcul de Neff
                    w_pos = np.maximum(wm_values, 0)
                    sum_w = np.sum(w_pos)

                    if sum_w > 0:
                        p_m = w_pos / sum_w
                        neff = 1.0 / np.sum(p_m**2)
                    else:
                        neff = 1.0

                    data.append({
                        "Layer": int(entry["layer"]),
                        "Fold": entry.get("fold", 0),
                        "Feature": entry["feature"],
                        "Neff": neff
                    })
                except:
                    continue

    df = pd.DataFrame(data)

    df['Feature Label'] = df['Feature'].map(rename_dict).fillna(df['Feature'])

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq." in label: return 'Frequency (Zipf)'
        if any(x in label for x in ["num.", "gender"]): return 'Morphology'
        return 'Other'

    df['Category'] = df['Feature Label'].apply(get_category)
    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    sns.set_context("paper")
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    category_order = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    for i, cat in enumerate(category_order):
        ax = axes[i]
        df_cat = df[df['Category'] == cat]

        if df_cat.empty:
            continue

        sns.lineplot(
            data=df_cat,
            x="Layer",
            y="Neff",
            hue="Feature Label",
            style="Feature Label",
            palette=color_map,
            markers=True,
            dashes=False,
            linewidth=2.5,
            ax=ax,
            err_style="band",
            errorbar=("ci", 95)
        )

        ax.axhline(1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_title(cat, fontweight='bold', size=18, pad=15)
        ax.set_xlabel("Layer", fontsize=18)

        ax.tick_params(axis='both', labelsize=14)

        if i == 0:
            ax.set_ylabel("Neff (Moyenne CV)", fontsize=18)
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
            fontsize=15,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0
        )
    plt.subplots_adjust(bottom=0.35, wspace=0.05)

    save_path = os.path.join(cfg.figures_dir, f"neff_full_CV_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_multi_layer_distributions(X, layers, n_components, n_nonzero, cfg):

    target_features = ["sentence_CLAUSE", "sentence_RC_attached", "subj_NUM"]
    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. number"
    }

    fig, axes = plt.subplots(3, 3, figsize=(24, 18), sharex=False, sharey=True)
    sns.set_style("white")

    for row_idx, layer in enumerate(layers):
        z_file_name = f"Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"
        z_path = os.path.join(cfg.z_cache, z_file_name)

        if not os.path.exists(z_path):
            print(f"Skipping layer {layer}: file not found")
            continue

        Z = np.load(z_path)

        log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")
        feature_to_top_atom = {}

        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    feat = entry['feature']
                    if feat in target_features:
                        importances = entry['permutation_importance']
                        top_atom_str = max(importances, key=lambda k: importances[k])
                        total_pos = sum(v for v in importances.values() if v > 0)
                        share = (importances[top_atom_str] / total_pos * 100) if total_pos > 0 else 0
                        feature_to_top_atom[feat] = (int(top_atom_str), share)

        for col_idx, feat in enumerate(target_features):
            ax = axes[row_idx, col_idx]

            if feat in feature_to_top_atom and feat in X.columns:
                atom_idx, share_pct = feature_to_top_atom[feat]

                df_plot = pd.DataFrame({
                    'Activation': Z[:, atom_idx],
                    'Value': X[feat].values
                })

                sns.histplot(
                    data=df_plot, x="Activation", hue="Value",
                    element="step", fill=True, stat="count", common_norm=False,
                    palette="tab10", alpha=0.4, ax=ax, linewidth=2,
                    legend=False,
                    bins=50
                )

                ax.set_yscale('log')

                if row_idx == 0:
                    feat_title = rename_dict[feat]
                    ax.set_title(f"{feat_title}", fontsize=30, fontweight='bold', pad=20)

                ax.text(0.95, 0.92, f"Atom {atom_idx}\nImp: {share_pct:.1f}%",
                        transform=ax.transAxes, fontsize=22, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer}\n\nCount (log)", fontsize=28, fontweight='bold')
            else:
                ax.set_ylabel("")

            if row_idx == 2:
                ax.set_xlabel("Activation intensity", fontsize=28, fontweight='bold')
            else:
                ax.set_xlabel("")

            ax.tick_params(axis='both', which='major', labelsize=24)
            sns.despine(ax=ax)

    plt.subplots_adjust(hspace=0.15, wspace=0.1)

    save_path = os.path.join(cfg.figures_dir, f"multi_layer_dist_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

def plot_all_features_distributions_log(X, layer, n_components, n_nonzero, cfg):
    z_file_name = f"Z_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.npy"
    z_path = os.path.join(cfg.z_cache, z_file_name)

    if not os.path.exists(z_path):
        print(f"Error: {z_file_name} not found")
        return

    Z = np.load(z_path)

    rename_dict = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. number",

        #"subj_GEN": "Subj. gender",
        #"subj_ZIPF": "Subj. frequency",
        #"obj_NUM": "Obj. number",
        #"obj_GEN": "Obj. gender",
        #"obj_ZIPF": "Obj. frequency",
        #"embed_NUM": "Embed. number",
        #"embed_GEN": "Embed. gender",
        #"embed_ZIPF": "Embed. frequency",
        #"verb_ZIPF": "Verb frequency",

    }

    log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")

    feature_to_top_atom = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                feat = entry['feature']
                if feat in rename_dict:
                    importances = entry['permutation_importance']
                    top_atom_str = max(importances, key=lambda k: importances[k])
                    top_atom = int(top_atom_str)

                    total_positive = sum(v for v in importances.values() if v > 0)
                    share_pct = (importances[top_atom_str] / total_positive * 100) if total_positive > 0 else 0.0

                    feature_to_top_atom[feat] = (top_atom, share_pct)

    features_to_plot = [f for f in rename_dict.keys() if f in X.columns]
    ncols = 3
    nrows = (len(features_to_plot) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
    axes = axes.flatten()

    sns.set_style("white")

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        if feat in feature_to_top_atom:
            atom_idx, share_pct = feature_to_top_atom[feat]

            df_plot = pd.DataFrame({
                'Activation': Z[:, atom_idx],
                'Value': X[feat].values
            })

            sns.histplot(
                data=df_plot, x="Activation", hue="Value",
                element="step", fill=True, stat="count", common_norm=False,
                palette="tab10", alpha=0.3, ax=ax, linewidth=2, legend=False,
                bins=50
            )

            ax.set_yscale('log')

            feat_name_bold = rename_dict[feat].replace(' ', '~')
            title_str = rf"$\mathbf{{{feat_name_bold}}}$ (Atom {atom_idx} - Imp: {share_pct:.1f}%)"
            ax.set_title(title_str, fontsize=20)

            if i % ncols == 0:
                ax.set_ylabel("Count (log)", fontsize=20)
            else:
                ax.set_ylabel("")

            if i >= len(features_to_plot) - ncols:
                ax.set_xlabel("Activation intensity", fontsize=20)
            else:
                ax.set_xlabel("")

            ax.tick_params(axis='both', which='major', labelsize=16)

            sns.despine(ax=ax, top=True, right=True)
        else:
            ax.axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Activation Distributions (Log Scale) - Layer {layer}", fontsize=28, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(cfg.figures_dir, f"dist_log_clean_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()





def plot_elbow(n_components, n_nonzero, cfg, layers = [0, 5, 11], name_feature = None):

    categories = ['Syntax', 'Morphology', 'Frequency (Zipf)']

    def get_category(label):
        if any(x in label for x in ["Relative", "Attachment"]): return 'Syntax'
        if "freq" in label.lower(): return 'Frequency (Zipf)'
        if any(x in label for x in ["num", "gender"]): return 'Morphology'
        return 'Other'

    fig, axes = plt.subplots(nrows=len(layers), ncols=len(categories),
                             figsize=(24, len(layers)*3 + 7), sharey=True, sharex=True)
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    for row_idx, layer in enumerate(layers):
        log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_components}_nnonzero{n_nonzero}.jsonl")

        feature_data = {}
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    raw_feat = entry['feature']
                    if name_feature and raw_feat not in name_feature:
                        continue
                    if raw_feat in rename_dict:
                        clean_feat = rename_dict[raw_feat]
                        importances = entry.get('permutation_importance', {})
                        pos_vals = sorted([v for v in importances.values() if v > 0], reverse=True)
                        total = sum(pos_vals)
                        if total > 0:
                            feature_data[clean_feat] = (np.cumsum(pos_vals) / total) * 100

        for col_idx, cat in enumerate(categories):
            ax = axes[row_idx, col_idx]
            feats_in_cat = [f for f in feature_data.keys() if get_category(f) == cat]

            for feat in feats_in_cat:
                cum_data = feature_data[feat]
                x_vals = np.arange(1, len(cum_data) + 1)
                ax.plot(x_vals, cum_data, marker='o', linestyle='-', linewidth=4,
                        markersize=8, label=feat, color=color_map.get(feat, "gray"))

            ax.set_xlim(0.8, 15.2)
            ax.set_ylim(0, 105)
            ax.set_xticks([1, 3, 6, 9, 12, 15])

            if row_idx == 0:
                ax.set_title(cat, fontsize=26, fontweight='bold', pad=25)

            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer}\n\nCumul. Importance", fontsize=26, fontweight='bold')

            if row_idx == len(layers) - 1:
                ax.set_xlabel("Number of Atoms", fontsize=26, fontweight='bold', labelpad=20)

                if feats_in_cat:
                    handles, labels = ax.get_legend_handles_labels()
                    num_items = len(labels)
                    num_cols = 2 if num_items > 3 else 1

                    ax.legend(
                        handles,
                        labels,
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.35),
                        ncol=num_cols,
                        fontsize=26,
                        frameon=False,
                        handletextpad=0.5,
                        columnspacing=1.0
                    )

            ax.tick_params(axis='both', which='major', labelsize=20)
            sns.despine(ax=ax)

    plt.subplots_adjust(bottom=0.20, top=0.92, hspace=0.15, wspace=0.08)

    save_path = os.path.join(cfg.figures_dir, f"elbow_grid_final_L{layer}_C{n_components}_K{n_nonzero}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

def plot_elbow_comparaison(cfg, layers=[0, 6, 11]):
    configs = [
        (64, 10, '#1f77b4'),  # Blue
        (64, 20, '#ff7f0e'),  # Orange
        (128, 10, '#2ca02c')  # Green
    ]

    target_features = {
        "sentence_CLAUSE": "Relative clause type",
        "sentence_RC_attached": "Attachment site",
        "subj_NUM": "Subj. number"
    }

    target_list = list(target_features.values())
    fig, axes = plt.subplots(nrows=len(layers), ncols=len(target_list),
                             figsize=(24, len(layers)*3 + 7), sharey=True, sharex=True)
    sns.set_style("whitegrid", {'grid.linestyle': ':'})

    for row_idx, layer in enumerate(layers):
        for n_comp, n_nonz, color_cfg in configs:
            log_path = os.path.join(cfg.log_dir, f"experiment_log_layer{layer}_ncomp{n_comp}_nnonzero{n_nonz}.jsonl")

            if not os.path.exists(log_path):
                continue

            feature_data = {}
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    raw_feat = entry['feature']
                    if raw_feat in target_features:
                        clean_feat = target_features[raw_feat]
                        importances = entry.get('permutation_importance', {})
                        pos_vals = sorted([v for v in importances.values() if v > 0], reverse=True)
                        total = sum(pos_vals)
                        if total > 0:
                            feature_data[clean_feat] = (np.cumsum(pos_vals) / total) * 100

            for col_idx, feat_clean in enumerate(target_list):
                ax = axes[row_idx, col_idx]
                if feat_clean in feature_data:
                    cum_sum = feature_data[feat_clean]
                    y_plot = cum_sum[:15]
                    x_plot = np.arange(1, len(y_plot) + 1)

                    label_str = f"C={n_comp}, K={n_nonz}" if (row_idx == 0 and col_idx == 0) else None

                    ax.plot(x_plot, y_plot, marker='o', linestyle='-', linewidth=4,
                            markersize=8, label=label_str, color=color_cfg, alpha=0.9)

                ax.set_xlim(0.8, 15.2)
                ax.set_ylim(0, 105)
                ax.set_xticks([1, 3, 6, 9, 12, 15])
                if row_idx == 0:
                    ax.set_title(feat_clean, fontsize=26, fontweight='bold', pad=25)

                if col_idx == 0:
                    ax.set_ylabel(f"Layer {layer}\n\nCumul. Importance", fontsize=26, fontweight='bold')

                if row_idx == len(layers) - 1:
                    ax.set_xlabel("Number of Atoms", fontsize=26, fontweight='bold', labelpad=20)

                    if col_idx == 1:
                        handles, labels = axes[0, 0].get_legend_handles_labels()
                        ax.legend(handles, labels, loc='upper center',
                                  bbox_to_anchor=(0.5, -0.35), ncol=3,
                                  fontsize=26, frameon=False, handletextpad=0.5)

                ax.tick_params(axis='both', which='major', labelsize=22)
                sns.despine(ax=ax)

    plt.subplots_adjust(bottom=0.20, top=0.92, hspace=0.15, wspace=0.08)
    save_path = os.path.join(cfg.figures_dir, f"elbow_comparison_CK_final.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

