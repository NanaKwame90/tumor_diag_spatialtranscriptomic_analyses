"""
Compare GAE architectures (MLP, GraphSAGE, GAT, SpaGCN) on the Mouse Brain Visium dataset.

Usage:
    python -m src.run_experiment          # from project root
    python src/run_experiment.py          # direct
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.transforms import RandomLinkSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import load_and_prepare
from src.models import GNNEncoder, MLPEncoder, SpaGCNEncoder, RNADecoder, TopologyDecoder, SpatialGAE
from src.train import train_epoch, evaluate

# ── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    "latent_dim": 64,
    "hidden_dim": 256,
    "num_layers": 2,
    "gat_heads": 4,       # must divide hidden_dim for intermediate GAT layers
    "mask_rate": 0.20,    # fraction of gene features masked per forward pass
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "rna_weight": 1.0,    # λ for RNA reconstruction loss
    "topo_weight": 1.0,   # λ for topology reconstruction loss
    "val_edge_ratio": 0.10,
    "test_edge_ratio": 0.10,
    "val_node_ratio": 0.15,
    "test_node_ratio": 0.15,
    "seed": 42,
    "log_every": 20,      # print progress every N epochs
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data splitting ────────────────────────────────────────────────────────────

def split_nodes(data):
    """Attach boolean train/val/test node masks to data."""
    rng = np.random.default_rng(CFG["seed"])
    n = data.num_nodes
    idx = rng.permutation(n)
    n_val = int(n * CFG["val_node_ratio"])
    n_test = int(n * CFG["test_node_ratio"])
    splits = {
        "val": idx[:n_val],
        "test": idx[n_val: n_val + n_test],
        "train": idx[n_val + n_test:],
    }
    for name, sel in splits.items():
        mask = torch.zeros(n, dtype=torch.bool)
        mask[sel] = True
        setattr(data, f"{name}_node_mask", mask)
    return data


def split_edges(data):
    """Use RandomLinkSplit to produce train/val/test edge sets.

    data.train_pos_edge_index  — edges used for message passing (excl. val/test)
    data.{val,test}_pos_edge_index / neg_edge_index  — supervision edges for AUROC
    """
    transform = RandomLinkSplit(
        num_val=CFG["val_edge_ratio"],
        num_test=CFG["test_edge_ratio"],
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        split_labels=True,
    )
    train_data, val_data, test_data = transform(data)
    data.train_pos_edge_index = train_data.edge_index
    data.val_pos_edge_index = val_data.pos_edge_label_index
    data.val_neg_edge_index = val_data.neg_edge_label_index
    data.test_pos_edge_index = test_data.pos_edge_label_index
    data.test_neg_edge_index = test_data.neg_edge_label_index
    return data


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(arch: str, num_genes: int) -> SpatialGAE:
    lat, hid, nl = CFG["latent_dim"], CFG["hidden_dim"], CFG["num_layers"]
    if arch == "mlp":
        encoder = MLPEncoder(num_genes, 2, hid, lat, nl)
    elif arch == "spagcn":
        encoder = SpaGCNEncoder(num_genes, hid, lat)
    else:
        encoder = GNNEncoder(num_genes, hid, lat, arch, nl, CFG["gat_heads"])
    return SpatialGAE(encoder, RNADecoder(lat, num_genes), TopologyDecoder())


# ── Training ──────────────────────────────────────────────────────────────────

def run(arch: str, data) -> tuple:
    """Train one architecture; returns (history_dict, test_mse, test_auroc)."""
    print(f"\n{'─' * 54}\n  Architecture: {arch.upper()}\n{'─' * 54}")
    model = build_model(arch, data.num_node_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    opt = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    history = {"val_mse": [], "val_auroc": [], "epoch": []}
    for epoch in range(1, CFG["epochs"] + 1):
        loss, rna_l, topo_l = train_epoch(
            model, data, opt,
            CFG["mask_rate"], CFG["rna_weight"], CFG["topo_weight"], DEVICE,
        )
        scheduler.step()

        if epoch % CFG["log_every"] == 0 or epoch == 1:
            val_mse, val_auroc = evaluate(
                model, data, "val", CFG["mask_rate"], DEVICE
            )
            history["val_mse"].append(val_mse)
            history["val_auroc"].append(val_auroc)
            history["epoch"].append(epoch)
            print(
                f"  [{epoch:3d}/{CFG['epochs']}]  loss={loss:.4f}  "
                f"rna={rna_l:.4f}  topo={topo_l:.4f}  "
                f"val_mse={val_mse:.4f}  val_auroc={val_auroc:.4f}"
            )

    test_mse, test_auroc = evaluate(model, data, "test", CFG["mask_rate"], DEVICE)
    print(f"  TEST → mse={test_mse:.4f}  auroc={test_auroc:.4f}")
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{arch}_weights.pt"))
    return history, test_mse, test_auroc


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_history: dict, results_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for arch, h in all_history.items():
        epochs = h["epoch"]
        axes[0, 0].plot(epochs, h["val_mse"], marker="o", markersize=3, label=arch.upper())
        axes[0, 1].plot(epochs, h["val_auroc"], marker="o", markersize=3, label=arch.upper())

    for ax, title, ylabel in [
        (axes[0, 0], "Validation RNA MSE ↓", "MSE"),
        (axes[0, 1], "Validation Edge AUROC ↑", "AUROC"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    archs = results_df.index.tolist()
    x = np.arange(len(archs))
    axes[1, 0].bar(x, results_df["Test MSE"], color="steelblue")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([a.upper() for a in archs])
    axes[1, 0].set_title("Test RNA MSE ↓")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(x, results_df["Test AUROC"], color="coral")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([a.upper() for a in archs])
    axes[1, 1].set_title("Test Edge AUROC ↑")
    axes[1, 1].set_ylabel("AUROC")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    print(f"Device: {DEVICE}")

    print("\n─── Loading & preparing data ───")
    data = load_and_prepare(
        n_top_genes=1500,
        n_neighs=5,
        save_fig_path=os.path.join(RESULTS_DIR, "spatial_heatmap.png"),
    )
    data = split_nodes(data)
    data = split_edges(data)
    print(
        f"\nSplits — nodes: train={data.train_node_mask.sum()}, "
        f"val={data.val_node_mask.sum()}, test={data.test_node_mask.sum()}"
    )
    print(
        f"Splits — edges: train_mp={data.train_pos_edge_index.size(1)}, "
        f"val+={data.val_pos_edge_index.size(1)}, "
        f"test+={data.test_pos_edge_index.size(1)}"
    )

    architectures = ["mlp", "sage", "gat", "spagcn"]
    all_history = {}
    rows = {}

    for arch in architectures:
        history, test_mse, test_auroc = run(arch, data)
        all_history[arch] = history
        rows[arch] = {"Test MSE": test_mse, "Test AUROC": test_auroc}

    df = pd.DataFrame(rows).T
    print(f"\n{'═' * 44}\n  Final Comparison\n{'═' * 44}")
    print(df.to_string(float_format="{:.4f}".format))
    df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
    plot_results(all_history, df)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
