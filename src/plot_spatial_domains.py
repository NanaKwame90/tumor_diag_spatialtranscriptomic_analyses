"""
Spatial domain clustering comparison: Ground Truth vs MLP, GraphSAGE, GAT, SpaGCN.

Uses the pre-computed 'cluster' annotations from the Visium HnE AnnData as
ground truth labels.  Falls back to PCA K-Means if annotations are absent.

Outputs:
    results/spatial_domain_clustering.png
    results/clustering_metrics.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import squidpy as sq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import load_and_prepare
from src.models import GNNEncoder, MLPEncoder, SpaGCNEncoder, RNADecoder, TopologyDecoder, SpatialGAE

CFG = {
    "latent_dim": 64,
    "hidden_dim": 256,
    "num_layers": 2,
    "gat_heads": 4,
    "seed": 42,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")


def get_ground_truth_labels(data_x: torch.Tensor):
    """Return (int_labels, n_clusters) aligned to the spatial crop used in data_prep."""
    adata = sq.datasets.visium_hne_adata()
    coords = adata.obsm["spatial"]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    mask = (
        (coords[:, 0] >= x_min + 0.25 * (x_max - x_min)) &
        (coords[:, 0] <= x_min + 0.75 * (x_max - x_min)) &
        (coords[:, 1] >= y_min + 0.25 * (y_max - y_min)) &
        (coords[:, 1] <= y_min + 0.75 * (y_max - y_min))
    )
    adata_crop = adata[mask]

    if "cluster" in adata_crop.obs.columns:
        labels = adata_crop.obs["cluster"].cat.codes.values
        n_clusters = int(adata_crop.obs["cluster"].nunique())
        print(f"Ground truth: dataset 'cluster' column → {n_clusters} unique spatial domains")
    else:
        # Fallback: K-Means on PCA of the already-preprocessed gene expression
        n_clusters = 7
        pca_z = PCA(n_components=30, random_state=CFG["seed"]).fit_transform(data_x.numpy())
        labels = KMeans(n_clusters=n_clusters, random_state=CFG["seed"], n_init=10).fit_predict(pca_z)
        print(f"No 'cluster' column; using PCA K-Means (K={n_clusters}) as pseudo ground truth")

    return labels, n_clusters


def build_model(arch: str, num_genes: int) -> SpatialGAE:
    lat, hid, nl = CFG["latent_dim"], CFG["hidden_dim"], CFG["num_layers"]
    if arch == "mlp":
        encoder = MLPEncoder(num_genes, 2, hid, lat, nl)
    elif arch == "spagcn":
        encoder = SpaGCNEncoder(num_genes, hid, lat)
    else:
        encoder = GNNEncoder(num_genes, hid, lat, arch, nl, CFG["gat_heads"])
    return SpatialGAE(encoder, RNADecoder(lat, num_genes), TopologyDecoder())


@torch.no_grad()
def get_latent(arch: str, data, num_genes: int) -> np.ndarray:
    model = build_model(arch, num_genes).to(DEVICE)
    weights_path = os.path.join(RESULTS_DIR, f"{arch}_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run  python src/run_experiment.py  first."
        )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    z = model.encode(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.pos.to(DEVICE))
    return z.cpu().numpy()


def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("─── Loading spatial transcriptomics data ───")
    data = load_and_prepare(n_top_genes=1500, n_neighs=5)
    gt_labels, n_clusters = get_ground_truth_labels(data.x)

    if len(gt_labels) != data.num_nodes:
        raise RuntimeError(
            f"Ground truth has {len(gt_labels)} labels but graph has {data.num_nodes} nodes. "
            "Mismatch in spatial crop boundaries."
        )

    architectures = ["mlp", "sage", "gat", "spagcn"]
    cluster_results = {}
    rows = []

    print(f"\n─── K-Means (K={n_clusters}) on latent embeddings ───")
    for arch in architectures:
        z = get_latent(arch, data, data.num_node_features)
        labels = KMeans(n_clusters=n_clusters, random_state=CFG["seed"], n_init=10).fit_predict(z)
        cluster_results[arch] = labels
        ari = adjusted_rand_score(gt_labels, labels)
        nmi = normalized_mutual_info_score(gt_labels, labels)
        rows.append({"Architecture": arch.upper(), "ARI": round(ari, 4), "NMI": round(nmi, 4)})
        print(f"  {arch.upper():10s}  ARI={ari:.4f}  NMI={nmi:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "clustering_metrics.csv"), index=False)

    # ── 5-panel spatial plot ──────────────────────────────────────────────────
    pos = data.pos.numpy()
    cmap = plt.get_cmap("tab10")
    dot_size = max(4, int(1500 / data.num_nodes * 12))  # scale dot size with density

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharex=True, sharey=True)

    def scatter(ax, labels, title):
        ax.scatter(
            pos[:, 0], pos[:, 1],
            c=labels, cmap=cmap, vmin=0, vmax=n_clusters - 1,
            s=dot_size, linewidths=0, alpha=0.9,
        )
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_aspect("equal")
        ax.axis("off")

    scatter(axes[0], gt_labels, "Ground Truth")
    for ax, arch in zip(axes[1:], architectures):
        m = df[df["Architecture"] == arch.upper()].iloc[0]
        scatter(
            ax, cluster_results[arch],
            f"{arch.upper()}\nARI={m['ARI']:.3f}  NMI={m['NMI']:.3f}",
        )

    plt.suptitle(
        "Spatial Domain Clustering — Ground Truth vs GAE Encoder Architectures",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "spatial_domain_clustering.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out}")
    print("\nFinal metrics:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
