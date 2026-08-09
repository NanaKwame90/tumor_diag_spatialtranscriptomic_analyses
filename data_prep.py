import numpy as np
import scipy.sparse as sp
import scanpy as sc
import squidpy as sq
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data


def load_and_prepare(n_top_genes: int = 1500, n_neighs: int = 5, save_fig_path: str = "spatial_heatmap.png") -> Data:
    # ── 1. Load ──────────────────────────────────────────────────────────────
    adata = sq.datasets.visium_hne_adata()
    print(f"Loaded: {adata.n_obs} spots × {adata.n_vars} genes")

    # ── 2. Spatial crop (~1000 spots) ────────────────────────────────────────
    coords = adata.obsm["spatial"]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    mask = (
        (coords[:, 0] >= x_min + 0.25 * (x_max - x_min)) &
        (coords[:, 0] <= x_min + 0.75 * (x_max - x_min)) &
        (coords[:, 1] >= y_min + 0.25 * (y_max - y_min)) &
        (coords[:, 1] <= y_min + 0.75 * (y_max - y_min))
    )
    adata = adata[mask].copy()
    print(f"After spatial crop: {adata.n_obs} spots")

    # ── 3. Normalise FIRST (required for flavor='seurat') ────────────────────
    adata.layers["counts"] = adata.X.copy()   # preserve raw counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ── 4. HVG selection ─────────────────────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"After HVG filter: {adata.n_obs} spots × {adata.n_vars} genes")

    # ── 5. 5-NN spatial graph ────────────────────────────────────────────────
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    # ── 6. Build PyG Data object ─────────────────────────────────────────────
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    x = torch.tensor(X, dtype=torch.float32)                        # [N, G]

    coords_crop = adata.obsm["spatial"].copy().astype(np.float32)
    coords_crop = (coords_crop - coords_crop.min(0)) / (coords_crop.ptp(0) + 1e-8)
    pos = torch.tensor(coords_crop, dtype=torch.float32)            # [N, 2]

    adj_coo = adata.obsp["spatial_connectivities"].tocoo()
    edge_index = torch.tensor(
        np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long     # [2, E]
    )

    data = Data(x=x, edge_index=edge_index, pos=pos)
    print(f"\nPyG Data: {data.num_nodes} nodes | {data.num_edges} edges | "
          f"x={tuple(data.x.shape)} | pos={tuple(data.pos.shape)}")

    # ── 7. Sanity checks ─────────────────────────────────────────────────────
    assert data.x.shape[0] == data.pos.shape[0]
    assert data.edge_index.max() < data.num_nodes
    assert not torch.isnan(data.x).any()
    print("Sanity checks passed ✓")

    # ── 8. Spatial heatmap ────────────────────────────────────────────────────
    mean_expr = x.mean(dim=1).numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sc_plot = ax.scatter(
        pos[:, 0].numpy(), pos[:, 1].numpy(),
        c=mean_expr, cmap="viridis", s=15, linewidths=0
    )
    plt.colorbar(sc_plot, ax=ax, label="Mean log-normalised expression")
    ax.set_title("Spatial expression heatmap")
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved → {save_fig_path}")

    return data


def plot_graph_topology(data, out_path: str = "results/spatial_graph_topology.png"):
    """Draw the 5-NN spatial graph: spots as nodes, edges as lines."""
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    # Draw edges first so nodes sit on top
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        ax.plot(
            [pos[src, 0], pos[dst, 0]],
            [pos[src, 1], pos[dst, 1]],
            color="#b0b8c8", linewidth=0.3, alpha=0.5, zorder=1,
        )
    ax.scatter(pos[:, 0], pos[:, 1], s=8, c="#2d6ea4", linewidths=0, zorder=2)
    ax.set_title(f"Spatial 5-NN Graph  ({data.num_nodes} spots, {data.num_edges} edges)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Graph topology plot saved → {out_path}")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    data = load_and_prepare()
    torch.save(data, "data/spatial_data.pt")
    print("Saved → data/spatial_data.pt")