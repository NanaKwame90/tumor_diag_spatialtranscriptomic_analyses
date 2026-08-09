# Tumor Diagnosis and Spatial Transcriptomics Analyses

A comparative study of Graph Autoencoder (GAE) architectures for self-supervised
representation learning on the 10x Visium Mouse Brain spatial transcriptomics dataset.

## Architectures Compared

| Encoder | Key Design | Params |
|---------|-----------|--------|
| **MLP** (baseline) | No message passing; concatenates expression + spatial coords | 498,716 |
| **GraphSAGE** | Mean-aggregation + self-concat; limits over-smoothing | 898,588 |
| **SpaGCN** | GCN with Gaussian spatial edge weights | 400,704 |
| **GAT** | 4-head attention; learns edge weights adaptively | 498,844 |

## Results

| Encoder | RNA MSE ↓ | Edge AUROC ↑ |
|---------|-----------|-------------|
| MLP (baseline) | **0.2863** | 0.9255 |
| GraphSAGE | 0.3171 | 0.9226 |
| SpaGCN | 0.3225 | 0.9380 |
| GAT | 0.3239 | **0.9544** |

Graph message passing improves topological edge reconstruction (GAT, AUROC = 0.9544)
but degrades per-spot gene expression imputation. The MLP achieves the lowest MSE
by processing each spot independently without neighbourhood smoothing.
SpaGCN offers a practical middle ground: competitive AUROC at the lowest parameter count.

## Project Structure

```
.
├── data_prep.py              # Dataset loading, preprocessing, 5-NN graph construction
├── requirements.txt
├── src/
│   ├── models.py             # GNNEncoder, SpaGCNEncoder, MLPEncoder, SpatialGAE
│   ├── train.py              # Dual-task training loop + standalone SpaGCN runner
│   ├── run_experiment.py     # Full experiment: split → train all 4 → plot results
│   └── plot_spatial_domains.py  # Spatial clustering visualisation
├── report/
│   ├── main.tex              # LaTeX report (memoir class, biblatex/biber)
│   └── bibliography.bib
├── data/                     # Downloaded/cached dataset (gitignored)
├── results/                  # Saved weights (.pt), comparison.png, metrics CSV
└── notebooks/
```

## Setup

```bash
conda create -n gnn python=3.12
conda activate gnn
pip install -r requirements.txt
pip install torch_geometric
```

## Usage

```bash
# 1. Preprocess data and build the spatial graph
python data_prep.py

# 2. Train all four architectures and generate comparison.png
python src/run_experiment.py

# 3. View results
open results/comparison.png
cat results/comparison.csv
```

## Self-Supervised Objectives

- **Gene expression reconstruction** — 20% of gene features randomly masked
  (Bernoulli) per forward pass; MSE computed on masked entries only.
- **Graph edge reconstruction** — 10% of edges held out as positive test edges;
  AUROC computed against an equal number of sampled negative edges.

## Dataset

10x Visium Mouse Brain H&E, loaded via [Squidpy](https://squidpy.readthedocs.io).
A 50×50% spatial crop yields **N = 862 spots**, **F = 1,500 highly variable genes**,
and **M = 4,310 undirected 5-NN edges**.
