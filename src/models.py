"""Graph Autoencoder architectures for spatial transcriptomics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GNNEncoder(nn.Module):
    """GNN encoder with configurable message-passing layers.

    gnn_type: 'gcn' | 'sage' | 'gat'
    GAT intermediate layers use multi-head concat; the final layer uses a single head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            is_last = i == num_layers - 1
            # For GAT hidden layers concat produces gat_heads × (hidden // gat_heads) = hidden_channels
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if is_last else hidden_channels

            if gnn_type == "gcn":
                self.convs.append(GCNConv(in_ch, out_ch))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(in_ch, out_ch))
            elif gnn_type == "gat":
                if is_last:
                    self.convs.append(
                        GATConv(in_ch, out_ch, heads=1, concat=False, dropout=dropout)
                    )
                else:
                    self.convs.append(
                        GATConv(
                            in_ch,
                            hidden_channels // gat_heads,
                            heads=gat_heads,
                            concat=True,
                            dropout=dropout,
                        )
                    )
            else:
                raise ValueError(
                    f"Unknown gnn_type '{gnn_type}'. Choose from: gcn, sage, gat"
                )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPEncoder(nn.Module):
    """Baseline MLP encoder (no message passing).

    Concatenates gene expression and spatial coordinates as input.
    """

    def __init__(
        self,
        gene_channels: int,
        coord_channels: int = 2,
        hidden_channels: int = 256,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_ch = gene_channels + coord_channels
        layers = []
        for i in range(num_layers):
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            layers.append(nn.Linear(in_ch, out_ch))
            if i < num_layers - 1:
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))
            in_ch = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, pos], dim=-1))


class SpaGCNEncoder(nn.Module):
    """GCN with Gaussian spatial edge weights, inspired by Hu et al. (2021) SpaGCN.

    w_ij = exp(-d_ij^2 / 2*sigma^2); sigma^2 estimated as mean squared edge distance.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        d2 = ((pos[edge_index[0]] - pos[edge_index[1]]) ** 2).sum(dim=1)
        edge_weight = torch.exp(-d2 / (2.0 * d2.mean() + 1e-8))
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight)


class RNADecoder(nn.Module):
    """Single linear layer to reconstruct gene expression from latent z."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class TopologyDecoder(nn.Module):
    """Dot-product decoder: score(i, j) = z_i · z_j (logit scale)."""

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class SpatialGAE(nn.Module):
    """Graph Autoencoder combining an encoder with RNA and topology decoders.

    Encoder can be a GNNEncoder (uses message passing) or MLPEncoder (ignores graph).
    """

    def __init__(
        self,
        encoder: nn.Module,
        rna_decoder: RNADecoder,
        topo_decoder: TopologyDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.rna_decoder = rna_decoder
        self.topo_decoder = topo_decoder
        self._is_mlp = isinstance(encoder, MLPEncoder)
        self._is_spagcn = isinstance(encoder, SpaGCNEncoder)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_mlp:
            return self.encoder(x, pos)
        if self._is_spagcn:
            return self.encoder(x, edge_index, pos)
        return self.encoder(x, edge_index)

    def decode_rna(self, z: torch.Tensor) -> torch.Tensor:
        return self.rna_decoder(z)

    def decode_topo(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.topo_decoder(z, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple:
        z = self.encode(x, edge_index, pos)
        return z, self.decode_rna(z)
