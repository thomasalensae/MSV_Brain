# ===========================
# Libraries
# ===========================
import os
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# ===========================
# Create timestamped folder for saving results
# ===========================
BASE_SAVE_DIR = r"C:\Users\thoma\Desktop\Code\Brain"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(BASE_SAVE_DIR, f"run_{timestamp}")
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"🗂️  All images and GIFs will be saved in: {SAVE_PATH}")

# ===========================
# Device configuration
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===========================
# Load BERT model and tokenizer
# ===========================
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()
model.to(device)
print("Number of BERT layers:", model.config.num_hidden_layers)

# ===========================
# Generate synthetic corpus
# ===========================
# Lexicons
subjects = [
    "The cat", "The young woman", "A student", "The scientist", "My friend",
    "This person", "The musician", "The director", "A traveler", "The professor",
    "Maxime", "A friend", "A woman", "A doctor", "A detective"
]

verbs = [
    "observes", "understands", "analyzes", "explores", "imagines", "prepares",
    "describes", "discovers", "repairs", "transforms", "is crazy about", "hates",
    "loves", "faces", "suspects"
]

objects = [
    "the world", "serenity", "tranquility", "the landscape", "a new idea",
    "an old manuscript", "the situation", "the problem", "a mechanism",
    "the hidden truth", "the system", "the machine", "the future", "the mystery",
    "memories", "emotions"
]

adverbs = [
    "carefully", "slowly", "quickly", "with care", "without hesitation", "sometimes",
    "often", "rarely", "silently", "joyfully", "sadly", "furiously", "emotionally",
    "anxiously", "lovingly"
]

emotion_words = ["sad", "happy", "anxious", "excited", "angry", "moved", "joyful", "grieved", "amazed"]
intensity_words = ["very", "gigantic", "enormous"]
animals = ["cat", "dog", "bird", "horse", "lion", "tiger", "fox", "rabbit", "elephant"]

question_starts = [
    "Why", "How", "When", "Where", "To what extent",
    "For what reasons", "At what moment", "In what way",
    "With whom", "Under what angle"
]

# Function to check presence of words in a sentence
def contains_words(sentence, words):
    sentence = sentence.lower()
    return int(any(w in sentence for w in words))

# Generate sentences
corpus = []

# Declarative sentences
for _ in range(2500):
    sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(adverbs)}."
    if random.random() < 0.3:  # Add emotion sometimes
        sentence += f" They feel {random.choice(emotion_words)}."
    if random.random() < 0.2:  # Add intensity sometimes
        sentence = f"{random.choice(intensity_words).capitalize()} {sentence}"
    if random.random() < 0.2:  # Add animal sometimes
        sentence += f" With a {random.choice(animals)}."
    corpus.append(sentence)

# Interrogative sentences
for _ in range(2500):
    question = f"{random.choice(question_starts)} {random.choice(subjects).lower()} {random.choice(verbs)} {random.choice(objects)} {random.choice(adverbs)}?"
    corpus.append(question)

# Shuffle the corpus
random.shuffle(corpus)

# ===========================
# Construct DataFrame with additional features
# ===========================
X = pd.DataFrame({
    "interrogative": [1 if "?" in s else 0 for s in corpus],
    "declarative": [0 if "?" in s else 1 for s in corpus],
    "emotion": [contains_words(s, emotion_words) for s in corpus],
    "intensity": [contains_words(s, intensity_words) for s in corpus],
    "animal": [contains_words(s, animals) for s in corpus],
})

print(X.head())

# ===========================
# Extract BERT embeddings for all layers
# ===========================
def get_all_bert_layers(sentences, batch_size=16):
    n_layers = model.config.num_hidden_layers + 1  # embeddings + 12 hidden layers
    layer_outputs = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            for layer_id in range(n_layers):
                cls_vecs = hidden_states[layer_id][:, 0, :]  # CLS token
                layer_outputs[layer_id].append(cls_vecs)
    
    return [torch.cat(x, dim=0) for x in layer_outputs]

layer_embeddings = get_all_bert_layers(corpus)
print(f"Number of layers retrieved: {len(layer_embeddings)}")

# ===========================
# Compute t-SNE for each layer
# ===========================
def compute_tsne_per_layer(layer_embeddings, n_components_pca=20):
    tsne_layers = []
    for layer_id, emb in enumerate(layer_embeddings):
        print(f"\n→ t-SNE for layer {layer_id}...")
        emb_np = emb.detach().cpu().numpy()
        pca = PCA(n_components=min(n_components_pca, emb_np.shape[1]))
        reduced = pca.fit_transform(emb_np)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random")
        tsne_emb = tsne.fit_transform(reduced)
        tsne_layers.append(tsne_emb)
    return tsne_layers

tsne_layers = compute_tsne_per_layer(layer_embeddings)

# ===========================
# Save t-SNE figures
# ===========================
all_figures = []

def plot_tsne_binary(tsne_emb, feature_values, feature_name, layer_id):
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=feature_values, cmap="coolwarm", alpha=0.85)
    plt.title(f"t-SNE — Layer {layer_id} — Feature: {feature_name}")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(label="0/1")

    layer_folder = os.path.join(SAVE_PATH, f"layer_{layer_id:02d}")
    os.makedirs(layer_folder, exist_ok=True)

    filename = os.path.join(layer_folder, f"tsne_{feature_name}.png")
    fig.savefig(filename, dpi=150)
    all_figures.append(fig)

for layer_id, tsne_emb in enumerate(tsne_layers):
    for feature in X.columns:
        vals = X[feature].values
        plot_tsne_binary(tsne_emb, vals, feature, layer_id)

print(f"\n🎉 All images have been saved in {SAVE_PATH}")

# ===========================
# Generate GIF per feature
# ===========================
def generate_feature_gif(feature_name, num_layers):
    gif_folder = os.path.join(SAVE_PATH, "gifs")
    os.makedirs(gif_folder, exist_ok=True)
    images = []

    for layer_id in range(num_layers):
        path = os.path.join(SAVE_PATH, f"layer_{layer_id:02d}", f"tsne_{feature_name}.png")
        if os.path.exists(path):
            images.append(imageio.v2.imread(path))

    if images:
        gif_path = os.path.join(gif_folder, f"tsne_{feature_name}.gif")
        imageio.mimsave(gif_path, images, duration=3)
        print(f"GIF generated for '{feature_name}' → {gif_path}")
    else:
        print(f"No images found for '{feature_name}', GIF not created.")

num_layers = len(tsne_layers)
print("\nGenerating GIFs...")
for feature in X.columns:
    generate_feature_gif(feature, num_layers)

# ===========================
# Show all figures (optional)
# ===========================
print("\nDisplaying all figures...")
for fig in all_figures:
    fig.show()
