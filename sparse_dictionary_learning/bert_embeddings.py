import os
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Transforms a sequence of token vectors produced by BERT into a single vector per sentence, making an average that ignores padding tokens"""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # [B, T, 1]
    summed = (hidden_states * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-9)                     # [B, 1]
    return summed / denom

@torch.no_grad()
def extract_layerwise_sentence_embeddings(sentences: list[str], model_name, batch_size = 32, max_length = 64, device = None, show_progress = True,) -> list[np.ndarray]:
    """Return a list of 12 numpy arrays (layer 1..12), each shaped [n_samples, hidden_dim]"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    n = len(sentences)
    n_batches = math.ceil(n / batch_size)

    # 12 layers (ignore embedding layer 0)
    buckets = [[] for _ in range(12)]

    iterator = range(0, n, batch_size)
    if show_progress:
        iterator = tqdm(list(iterator), desc="BERT batches")

    for start in iterator:

        # Tokenization of sentences
        batch = sentences[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

        # Passage in BERT
        out = model(**enc)
        hs = out.hidden_states

        # Pooling as a phrase representation
        for layer_idx in range(1, 13):
            pooled = mean_pool(hs[layer_idx], enc["attention_mask"])  # [B, H]
            buckets[layer_idx - 1].append(pooled.detach().cpu().float())

    Y_layers = []
    for l in range(12):
        Y_layers.append(torch.cat(buckets[l], dim=0).numpy())
    return Y_layers

def cache_path_for_embeddings(cfg, df) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(cfg.dataset_csv))[0]
    return os.path.join(
        cfg.cache_dir,
        f"{base}__{cfg.model_name.replace('/', '_')}__maxlen{cfg.max_length}__bs{cfg.batch_size}__n{len(df)}__str{cfg.stratify_col}.npy"
    )

def compute_embeddings(df, cfg) -> list[np.ndarray]:
    """Compute or load cached BERT embeddings for sentences in df."""

    # Load cached BERT embeddings
    cache_path = cache_path_for_embeddings(cfg, df)
    print(cache_path)
    if os.path.exists(cache_path):
        print("Loading cached embeddings:", cache_path)
        Y_stack = np.load(cache_path)  # shape: [n, 12, hidden]
        Y_layers = [Y_stack[:, i, :] for i in range(Y_stack.shape[1])]

    # Compute BERT embeddings
    else:
        sentences = df[cfg.sentence_col].astype(str).tolist()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Computing embeddings on device:", device)
        Y_layers = extract_layerwise_sentence_embeddings(
            sentences,
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            device=device,
            show_progress=True,
        )
        Y_stack = np.stack(Y_layers, axis=1)  # [n, 12, hidden]
        np.save(cache_path, Y_stack)
        print("Saved cache_categ_1:", cache_path)

    print("Layer shapes:", [y.shape for y in Y_layers])
    return Y_layers
