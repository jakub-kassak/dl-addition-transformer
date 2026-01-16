import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import GPTLightningModule
import os


def main():
    parser = argparse.ArgumentParser(description="PCA Analysis of Token Embeddings")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Dimensions for PCA (1, 2, or 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pca_visualization.png",
        help="Output file for the plot",
    )
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model from {args.ckpt}...")
    model = GPTLightningModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    # 2. Extract Embeddings
    # token_embedding_table is (vocab_size, n_embd)
    embeddings = model.token_embedding_table.weight.detach().cpu().numpy()
    vocab_size = embeddings.shape[0]

    # 3. Define Labels
    # Based on data.py: self.chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
    # Total 24 tokens
    if vocab_size == 24:
        labels = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
    else:
        print(f"Warning: Unexpected vocab size {vocab_size}. Using generic labels.")
        labels = [f"token_{i}" for i in range(vocab_size)]

    # 4. Perform PCA
    print(f"Performing PCA to {args.dims} dimensions...")
    pca = PCA(n_components=args.dims)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 5. Visualize
    print(f"Visualizing in {args.dims}D...")
    fig = plt.figure(figsize=(10, 8))

    if args.dims == 1:
        ax = fig.add_subplot(111)
        ax.scatter(
            reduced_embeddings[:, 0], np.zeros_like(reduced_embeddings[:, 0]), alpha=0.5
        )
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (reduced_embeddings[i, 0], 0),
                rotation=90,
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        ax.get_yaxis().set_visible(False)
        ax.set_title("PCA of Token Embeddings (1D)")

    elif args.dims == 2:
        ax = fig.add_subplot(111)
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
        for i, label in enumerate(labels):
            ax.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA of Token Embeddings (2D)")
        ax.grid(True, linestyle="--", alpha=0.7)

    elif args.dims == 3:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            alpha=0.5,
        )
        for i, label in enumerate(labels):
            ax.text(
                reduced_embeddings[i, 0],
                reduced_embeddings[i, 1],
                reduced_embeddings[i, 2],
                label,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA of Token Embeddings (3D)")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved visualization to {args.output}")

    # Try to show it if in an interactive environment, but we are an agent
    # plt.show()


if __name__ == "__main__":
    main()
