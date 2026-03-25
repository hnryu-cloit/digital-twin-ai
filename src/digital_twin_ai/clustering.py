"""
clustering.py - Logic for persona clustering using UMAP and K-Means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from typing import List


class ClusteringManager:
    """Class for managing dimensionality reduction and clustering."""

    def __init__(self, n_personas: int = 7, random_state: int = 42):
        self.n_personas = n_personas
        self.random_state = random_state
        self.reducer = umap.UMAP(
            n_components=2, 
            random_state=random_state, 
            n_neighbors=30, 
            min_dist=0.1
        )

    def reduce_dimensions(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Dimensionality reduction using UMAP."""
        numeric_df = df[feature_cols].select_dtypes(include=[np.number])
        print(f"Using {len(numeric_df.columns)} numeric features for UMAP.")
        
        print("Reducing dimensions using UMAP...")
        return self.reducer.fit_transform(numeric_df.values)

    def cluster(self, embedding: np.ndarray) -> np.ndarray:
        """Perform K-Means clustering."""
        n_samples = len(embedding)
        if n_samples < 2:
            raise ValueError("At least 2 samples are required for clustering.")
        n_clusters = min(self.n_personas, n_samples)
        if n_clusters == n_samples:
            n_clusters = max(2, n_samples - 1)
        print(f"K-Means clustering (k={n_clusters})...")
        km = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = km.fit_predict(embedding)
        if n_clusters > 1 and n_samples > n_clusters:
            sample_size = min(2000, n_samples)
            score = silhouette_score(embedding, labels, sample_size=sample_size)
            print(f"Silhouette Score: {score:.3f}")
        return labels

    def visualize_clusters(self, embedding: np.ndarray, labels: np.ndarray, output_path: str) -> None:
        """Visualize clusters and save the plot."""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6)
        plt.colorbar(scatter, label="Cluster")
        plt.title("Persona Clusters (UMAP)")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Cluster visualization saved to: {output_path}")

    def find_optimal_k(self, embedding: np.ndarray, k_range: range, output_path: str) -> None:
        """Find optimal k using Elbow and Silhouette methods."""
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(embedding)
            inertias.append(km.inertia_)
            sample_size = min(2000, len(embedding))
            silhouettes.append(silhouette_score(embedding, labels, sample_size=sample_size))
            print(f"  k={k} | inertia={km.inertia_:.0f} | silhouette={silhouettes[-1]:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(k_range, inertias, "o-")
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inertia")
        axes[1].plot(k_range, silhouettes, "o-")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("k")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Elbow graph saved to: {output_path}")
