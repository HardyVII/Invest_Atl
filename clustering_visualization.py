# K-means clustering with Elbow Method & t-SNE visualization

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Load sentiment-labeled data
uber_df = pd.read_csv('data/sentiment/UBER_Sentiment.csv')
marta_df = pd.read_csv('data/sentiment/Marta_Sentiment.csv')
beltline_df = pd.read_csv('data/sentiment/AtlBeltline_Sentiment.csv')

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create output directories if not exist
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Function to apply Elbow Method
def plot_elbow(embeddings, org_name, max_k=10):
    distortions = []
    print(f"📐 Running Elbow Method for {org_name}...")

    for k in tqdm(range(2, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title(f'Elbow Method for {org_name}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{org_name}_elbow.png')
    plt.show()
    print(f"📊 Elbow plot saved as outputs/plots/{org_name}_elbow.png\n")

# Function for clustering and visualization
def cluster_and_visualize(df, org_name, k):
    print(f"📌 Clustering & visualizing: {org_name}")

    # Generate embeddings
    texts = df['post_body_text'].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Run Elbow Method
    plot_elbow(embeddings, org_name)

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df['cluster'] = labels

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(f"{org_name} - K-Means Clusters on Safety-related Posts")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{org_name}_clusters.png')
    plt.show()

    # Save clustered data
    df.to_csv(f'data/processed/{org_name}_Clustered.csv', index=False)
    print(f"✅ Saved clustered data and cluster plot for {org_name}\n")

# Plot the elbow
UBER_texts = uber_df['post_body_text'].astype(str).tolist()
embeddings = model.encode(UBER_texts, show_progress_bar=True)
plot_elbow(embeddings, org_name="UBER")

Marta_texts = marta_df['post_body_text'].astype(str).tolist()
embeddings = model.encode(Marta_texts, show_progress_bar=True)
plot_elbow(embeddings, org_name="MARTA")

AtlBeltline_texts = beltline_df['post_body_text'].astype(str).tolist()
embeddings = model.encode(AtlBeltline_texts, show_progress_bar=True)
plot_elbow(embeddings, org_name="ATLBELTLINE")

# Hardcode the best k based on the elbow plot
cluster_and_visualize(uber_df, "UBER", k=3)
cluster_and_visualize(marta_df, "MARTA", k=3)
cluster_and_visualize(beltline_df, "AtlantaBeltline", k=3)
