"""
Generate two publication-quality figures:

  Figure 1 — Spatial heatmaps (2 panels):
      Left  : Mean log-normalised gene expression per spot (RNA-seq).
      Right : Mean Gaussian edge weight per spot (5-NN spatial topology).

  Figure 2 — Reconstruction comparison (5 panels):
      Ground Truth | MLP | GraphSAGE | GAT | SpaGCN
      Each panel shows each model's reconstructed mean expression,
      visualising how well message-passing recovers the spatial pattern.

Run after src/run_experiment.py (weights must exist in results/).
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import load_and_prepare
from src.models import GNNEncoder, MLPEncoder, SpaGCNEncoder, RNADecoder, TopologyDecoder, SpatialGAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {"latent_dim": 64, "hidden_dim": 256, "num_layers": 2, "gat_heads": 4}

ARCH_LABELS = {
    "mlp":    "MLP (baseline)",
    "sage":   "GraphSAGE",
    "gat":    "GAT",
    "spagcn": "SpaGCN",
}


# ── Model helpers ─────────────────────────────────────────────────────────────

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
def get_reconstruction(arch: str, data) -> np.ndarray:
    """Return per-spot mean reconstructed expression for one architecture."""
    model = build_model(arch, data.num_node_features).to(DEVICE)
    weights = os.path.join(RESULTS_DIR, f"{arch}_weights.pt")
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    _, x_recon = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.pos.to(DEVICE))
    return x_recon.cpu().numpy().mean(axis=1)   # [N]


# ── Topology weight per spot ───────────────────────────────────────────────────

def gaussian_edge_weights(data) -> np.ndarray:
    """Mean Gaussian edge weight per spot: w_ij = exp(-d²/2σ²)."""
    pos = data.pos                                          # [N, 2]
    ei = data.edge_index                                    # [2, E]
    d2 = ((pos[ei[0]] - pos[ei[1]]) ** 2).sum(dim=1)
    w = torch.exp(-d2 / (2.0 * d2.mean() + 1e-8))         # [E]
    # Accumulate per-node mean weight
    node_w = torch.zeros(data.num_nodes)
    node_count = torch.zeros(data.num_nodes)
    node_w.scatter_add_(0, ei[0], w)
    node_count.scatter_add_(0, ei[0], torch.ones(w.size(0)))
    return (node_w / node_count.clamp(min=1)).numpy()      # [N]


# ── Figure 1: dual spatial heatmap ────────────────────────────────────────────

def plot_spatial_heatmaps(data, out_path: str):
    pos = data.pos.numpy()
    expr = data.x.numpy().mean(axis=1)
    topo = gaussian_edge_weights(data)
    dot  = max(6, int(2000 / data.num_nodes * 10))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A — RNA-seq expression
    sc0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=expr,
                          cmap="viridis", s=dot, linewidths=0)
    axes[0].set_title("Spatial Gene Expression\n(mean log-normalised, RNA-seq)",
                      fontsize=11, fontweight="bold")
    plt.colorbar(sc0, ax=axes[0], label="Mean expression")

    # Panel B — spatial topology
    sc1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=topo,
                          cmap="plasma", s=dot, linewidths=0)
    axes[1].set_title("Spatial Graph Topology\n(mean Gaussian edge weight, 5-NN)",
                      fontsize=11, fontweight="bold")
    plt.colorbar(sc1, ax=axes[1], label="Mean edge weight  w = exp(−d²/2σ²)")

    for ax in axes:
        ax.set_xlabel("x (normalised)")
        ax.set_ylabel("y (normalised)")
        ax.set_aspect("equal")

    plt.suptitle("Spatial Heatmaps — Expression and Topology", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Figure 2: ground truth vs model reconstructions ──────────────────────────

def plot_reconstruction_comparison(data, architectures, out_path: str):
    pos = data.pos.numpy()
    dot = max(6, int(2000 / data.num_nodes * 10))

    n_panels = 1 + len(architectures)   # GT + 4 models
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5),
                             sharex=True, sharey=True)

    gt_expr = data.x.numpy().mean(axis=1)
    vmin, vmax = gt_expr.min(), gt_expr.max()

    # Ground truth panel
    sc = axes[0].scatter(pos[:, 0], pos[:, 1], c=gt_expr,
                         cmap="viridis", s=dot, linewidths=0,
                         vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth\n(actual expression)", fontsize=9.5,
                      fontweight="bold")

    # One panel per architecture
    for ax, arch in zip(axes[1:], architectures):
        recon = get_reconstruction(arch, data)
        # Rescale reconstructed values into ground-truth range for fair comparison
        recon = (recon - recon.min()) / (recon.ptp() + 1e-8) * (vmax - vmin) + vmin
        ax.scatter(pos[:, 0], pos[:, 1], c=recon,
                   cmap="viridis", s=dot, linewidths=0,
                   vmin=vmin, vmax=vmax)
        ax.set_title(f"{ARCH_LABELS[arch]}\n(reconstructed)", fontsize=9.5,
                     fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    ColorbarBase(cbar_ax, cmap=plt.get_cmap("viridis"),
                 norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                 label="Mean log-normalised expression")

    plt.suptitle(
        "Spatial Expression Reconstruction — Ground Truth vs GAE Encoders",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Figure 3: edge topology reconstruction comparison ─────────────────────────

@torch.no_grad()
def get_edge_scores(arch: str, data) -> np.ndarray:
    """Per-edge predicted probability σ(z_i·z_j) for all 5-NN edges."""
    model = build_model(arch, data.num_node_features).to(DEVICE)
    weights = os.path.join(RESULTS_DIR, f"{arch}_weights.pt")
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    z  = model.encode(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.pos.to(DEVICE))
    ei = data.edge_index.to(DEVICE)
    return model.decode_topo(z, ei).sigmoid().cpu().numpy()   # [E]


@torch.no_grad()
def get_neg_edge_scores(arch: str, data, neg_ei: torch.Tensor) -> np.ndarray:
    """Per-edge predicted probability for sampled negative (non-existing) edges."""
    model = build_model(arch, data.num_node_features).to(DEVICE)
    weights = os.path.join(RESULTS_DIR, f"{arch}_weights.pt")
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    z = model.encode(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.pos.to(DEVICE))
    return model.decode_topo(z, neg_ei.to(DEVICE)).sigmoid().cpu().numpy()


def _sample_negative_edges(data, n_samples: int, seed: int = 42) -> torch.Tensor:
    """Sample n_samples node pairs that are NOT in the 5-NN graph."""
    rng = np.random.default_rng(seed)
    pos_set = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    N = data.num_nodes
    neg = []
    while len(neg) < n_samples:
        batch = rng.integers(0, N, size=(n_samples * 3, 2))
        for u, v in batch:
            if u != v and (int(u), int(v)) not in pos_set:
                neg.append([u, v])
            if len(neg) == n_samples:
                break
    arr = np.array(neg[:n_samples], dtype=np.int64).T   # [2, n_samples]
    return torch.tensor(arr, dtype=torch.long)


def plot_topology_comparison(data, architectures, out_path: str):
    """5-panel: Ground Truth + per-model edges (true=green, false=red) coloured by
    predicted probability, revealing each architecture's edge discrimination spatially."""
    pos    = data.pos.numpy()
    ei_pos = data.edge_index.numpy()                    # true edges
    n_neg  = ei_pos.shape[1]                            # match true edge count
    neg_ei = _sample_negative_edges(data, n_neg)
    ei_neg = neg_ei.numpy()
    dot    = max(5, int(1800 / data.num_nodes * 10))

    n_panels = 1 + len(architectures)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5),
                             sharex=True, sharey=True)

    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def draw_all_edges(ax, pos_scores, neg_scores):
        # Negative edges first (behind), then positive on top
        for k in range(ei_neg.shape[1]):
            s, d = ei_neg[0, k], ei_neg[1, k]
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    color=cmap(norm(neg_scores[k])), lw=0.35, alpha=0.5, zorder=1)
        for k in range(ei_pos.shape[1]):
            s, d = ei_pos[0, k], ei_pos[1, k]
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    color=cmap(norm(pos_scores[k])), lw=0.5, alpha=0.7, zorder=2)

    # Ground truth panel: true edges solid blue, negative edges faint grey
    ax0 = axes[0]
    for k in range(ei_neg.shape[1]):
        s, d = ei_neg[0, k], ei_neg[1, k]
        ax0.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                 color="#cccccc", lw=0.25, alpha=0.3, zorder=1)
    for k in range(ei_pos.shape[1]):
        s, d = ei_pos[0, k], ei_pos[1, k]
        ax0.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                 color="#1a6fbf", lw=0.5, alpha=0.6, zorder=2)
    ax0.scatter(pos[:, 0], pos[:, 1], s=dot, c="#1a3a5c", linewidths=0, zorder=3)
    ax0.set_title("Ground Truth\n(blue=true edge, grey=non-edge)",
                  fontsize=9.5, fontweight="bold")

    # Model panels: colour edges by predicted probability (green=high, red=low)
    for ax, arch in zip(axes[1:], architectures):
        pos_scores = get_edge_scores(arch, data)
        neg_scores = get_neg_edge_scores(arch, data, neg_ei)
        draw_all_edges(ax, pos_scores, neg_scores)
        ax.scatter(pos[:, 0], pos[:, 1], s=dot, c="#1a3a5c", linewidths=0, zorder=3)
        ax.set_title(f"{ARCH_LABELS[arch]}\n(green=high prob, red=low prob)",
                     fontsize=9.5, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Predicted edge probability  σ(z_i · z_j)")

    plt.suptitle(
        "Spatial Edge Topology Reconstruction — Ground Truth vs GAE Encoders\n"
        "True edges (5-NN) and equal-sized negative sample coloured by predicted probability",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("─── Loading data ───")
    data = load_and_prepare(n_top_genes=1500, n_neighs=5)

    architectures = ["mlp", "sage", "gat", "spagcn"]

    print("\n─── Figure 1: spatial heatmaps ───")
    plot_spatial_heatmaps(
        data,
        out_path=os.path.join(RESULTS_DIR, "spatial_heatmaps.png"),
    )

    print("\n─── Figure 2: expression reconstruction comparison ───")
    plot_reconstruction_comparison(
        data, architectures,
        out_path=os.path.join(RESULTS_DIR, "reconstruction_comparison.png"),
    )

    print("\n─── Figure 3: edge topology reconstruction comparison ───")
    plot_topology_comparison(
        data, architectures,
        out_path=os.path.join(RESULTS_DIR, "topology_comparison.png"),
    )


if __name__ == "__main__":
    main()
