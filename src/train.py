"""Training loop and evaluation utilities for the spatial GAE.

Run directly to train SpaGCN independently and print test results:
    python src/train.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


def mask_features(x: torch.Tensor, mask_rate: float = 0.20):
    """Zero-out mask_rate fraction of features; returns (x_masked, bool_mask)."""
    mask = torch.rand_like(x) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def train_epoch(
    model,
    data,
    optimizer,
    mask_rate: float = 0.20,
    rna_weight: float = 1.0,
    topo_weight: float = 1.0,
    device: str = "cpu",
) -> tuple:
    """One training step; returns (total_loss, rna_loss, topo_loss)."""
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    pos = data.pos.to(device)
    train_edge = data.train_pos_edge_index.to(device)

    x_masked, feat_mask = mask_features(x, mask_rate)
    z, x_recon = model(x_masked, train_edge, pos)

    # RNA loss computed only on the randomly masked entries
    rna_loss = F.mse_loss(x_recon[feat_mask], x[feat_mask])

    # Topology loss: observed train edges vs equal number of sampled negatives
    neg_edge = negative_sampling(
        train_edge, num_nodes=data.num_nodes, num_neg_samples=train_edge.size(1)
    ).to(device)
    pos_scores = model.decode_topo(z, train_edge)
    neg_scores = model.decode_topo(z, neg_edge)
    topo_labels = torch.cat([
        torch.ones(pos_scores.size(0), device=device),
        torch.zeros(neg_scores.size(0), device=device),
    ])
    topo_loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]), topo_labels
    )

    loss = rna_weight * rna_loss + topo_weight * topo_loss
    loss.backward()
    optimizer.step()
    return loss.item(), rna_loss.item(), topo_loss.item()


@torch.no_grad()
def evaluate(
    model,
    data,
    split: str = "val",
    mask_rate: float = 0.20,
    device: str = "cpu",
) -> tuple:
    """Returns (RNA MSE on masked+split-node entries, edge AUROC for the split).

    Feature masking uses a fixed seed so evaluation is deterministic.
    """
    model.eval()
    x = data.x.to(device)
    pos = data.pos.to(device)
    train_edge = data.train_pos_edge_index.to(device)

    # Deterministic masking: save and restore RNG state
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    x_masked, feat_mask = mask_features(x, mask_rate)
    torch.set_rng_state(rng_state)

    z, x_recon = model(x_masked, train_edge, pos)

    # RNA MSE: masked features of held-out nodes only
    node_mask = getattr(data, f"{split}_node_mask").to(device)
    combined = feat_mask & node_mask.unsqueeze(-1)
    rna_mse = F.mse_loss(x_recon[combined], x[combined]).item()

    # Edge AUROC: held-out positive edges vs pre-sampled negatives
    pos_edge = getattr(data, f"{split}_pos_edge_index").to(device)
    neg_edge = getattr(data, f"{split}_neg_edge_index").to(device)
    scores = torch.cat([
        model.decode_topo(z, pos_edge).sigmoid(),
        model.decode_topo(z, neg_edge).sigmoid(),
    ]).cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_edge.size(1)),
        np.zeros(neg_edge.size(1)),
    ])
    auroc = roc_auc_score(labels, scores)
    return rna_mse, auroc


if __name__ == "__main__":
    # Standalone runner: trains SpaGCN independently and prints test results.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_prep import load_and_prepare
    from src.models import SpaGCNEncoder, RNADecoder, TopologyDecoder, SpatialGAE
    from src.run_experiment import split_nodes, split_edges, CFG, RESULTS_DIR

    torch.manual_seed(42)
    np.random.seed(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    data = load_and_prepare(n_top_genes=1500, n_neighs=5)
    data = split_nodes(data)
    data = split_edges(data)

    G = data.num_node_features
    lat, hid = CFG["latent_dim"], CFG["hidden_dim"]

    encoder = SpaGCNEncoder(G, hid, lat)
    print(f"\n{'─'*50}\n  SPAGCN ({sum(p.numel() for p in encoder.parameters()):,} params)\n{'─'*50}")

    model = SpatialGAE(encoder, RNADecoder(lat, G), TopologyDecoder()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    for ep in range(1, CFG["epochs"] + 1):
        train_epoch(model, data, opt, CFG["mask_rate"], 1.0, 1.0, DEVICE)
        sched.step()
        if ep % 50 == 0:
            val_mse, val_auroc = evaluate(model, data, "val", CFG["mask_rate"], DEVICE)
            print(f"  [{ep:3d}/{CFG['epochs']}]  val_mse={val_mse:.4f}  val_auroc={val_auroc:.4f}")

    test_mse, test_auroc = evaluate(model, data, "test", CFG["mask_rate"], DEVICE)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "spagcn_weights.pt"))
    print(f"\n  TEST → mse={test_mse:.4f}  auroc={test_auroc:.4f}")
