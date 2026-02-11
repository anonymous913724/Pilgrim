# TASER-TGNN

# The original TASER-TGNN is for temporal interaction graphs (events). It expects
# "temporal blocks" that include neighbor timestamps and (optionally) edge features.
# PEMS-BAY dataset is snapshot-based (StaticGraphTemporalSignal / PGT snapshots),
# with edge_index and per-node features each time.

# This adapter:
# Builds a "block" per call from a static PyG edge_index using 1-hop neighbor sampling.
# Provides required tensors: root_node_feature, neighbor_node_feature, neighbor_edge_feature,
#  root_ts, neighbor_ts.
# Uses TASER's TransformerAggregator forward to produce node embeddings [N, out_channels].

# It implements the DCRNN-like step-wise signature:
#    H = cell(x_t, edge_index, edge_weight, H_prev)

#Notes:
# Timestamps: for PEMS we don't have meaningful per-edge event times. We set all ts to 0.
# Edge features: we use edge_weight if provided, else 1.0.
# Neighbor sampling: fixed B neighbors per node. This is fast and keeps memory bounded.

# Must ensure TASER source is importable:
# Place extracted taser code so that one of these exists:
#    ./taser/src/model.py (or ./taser-tgnn/src/model.py)
# This adapter will add ./taser/src and ./taser-tgnn/src to sys.path automatically.


import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

def _maybe_add_taser_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [
        os.path.join(here, "taser", "src"),
        os.path.join(here, "taser-tgnn", "src"),
        os.path.join(os.getcwd(), "taser", "src"),
        os.path.join(os.getcwd(), "taser-tgnn", "src"),
    ]
    for p in cand:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_maybe_add_taser_to_path()

# Import TASER components
try:
    # model.py defines TransformerAggregator
    from model import TransformerAggregator
except Exception:
    # or layers.py
    try:
        from layers import TransformerAggregator
    except Exception as e:
        raise ImportError(
            "Cannot import TASER TransformerAggregator. "
            "Make sure taser source is extracted under ./taser/src or ./taser-tgnn/src "
            "and contains model.py or layers.py with TransformerAggregator."
        ) from e

@dataclass
class _StaticBlock:
    # Root features
    root_node_feature: torch.Tensor        # [N, F]
    # Neighbor features
    neighbor_node_feature: torch.Tensor    # [N, B, F]
    neighbor_edge_feature: torch.Tensor    # [N, B, 1]
    # Times (for PEMS: all zeros)
    root_ts: torch.Tensor                  # [N]
    neighbor_ts: torch.Tensor              # [N, B]
    # sizes
    n: int
    b: int


def _build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int):
    """
    Build adjacency lists for incoming neighbors (dst gets src).
    edge_index: [2, E]
    Returns: list[list[int]] of length N
    """
    src = edge_index[0].detach().cpu()
    dst = edge_index[1].detach().cpu()
    nbrs = [[] for _ in range(num_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        nbrs[d].append(s)
    return nbrs


class TaserTGNNCell(nn.Module):
    """
    TASER cell adapter

    forward(x_t, edge_index, edge_weight=None, H_prev=None) -> H_t (node embeddings)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int = 16,
        dim_time: int = 16,
        att_head: int = 4,
        dropout: float = 0.0,
        time_enc: str = "learnable",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_neighbors = int(num_neighbors)
        
        self.layer = TransformerAggregator(
            dim_node_feat=in_channels,
            dim_edge_feat=1,
            dim_time=dim_time,
            num_head=att_head,
            dim_out=out_channels,
            dropout=dropout,
            time_encoder_type=time_enc,
        )

        self._nbrs = None
        self._w_dense = None

    def _prepare_edge_weight_dense(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        device: torch.device,
    ):
        """
        Build a dict mapping (dst -> list of (src, weight)).
        This is used to fill neighbor_edge_feature.
        """
        src = edge_index[0]
        dst = edge_index[1]

        if edge_weight is None:
            ew = torch.ones(src.numel(), device=device, dtype=torch.float32)
        else:
            ew = edge_weight.to(device=device, dtype=torch.float32)

        # build per-dst arrays
        w_lists = [[] for _ in range(num_nodes)]
        for i in range(src.numel()):
            d = int(dst[i].item())
            w_lists[d].append(float(ew[i].item()))
        return w_lists

    def _build_block(
        self,
        x_t: torch.Tensor,                # [N, F]
        edge_index: torch.Tensor,         # [2, E]
        edge_weight: Optional[torch.Tensor],
    ) -> _StaticBlock:
        device = x_t.device
        N = x_t.size(0)
        B = self.num_neighbors

        # cache neighbor lists for a fixed graph size
        if (self._nbrs is None) or (len(self._nbrs) != N):
            self._nbrs = _build_neighbor_lists(edge_index, N)

        # cache per-dst edge weights
        w_lists = self._prepare_edge_weight_dense(edge_index, edge_weight, N, device)

        # build neighbor indices [N,B]
        # if a node has <B neighbors, pad with itself (self-loop)
        neigh_idx = torch.empty((N, B), device=device, dtype=torch.long)
        neigh_w = torch.empty((N, B, 1), device=device, dtype=torch.float32)

        for v in range(N):
            nbrs = self._nbrs[v]
            if len(nbrs) == 0:
                idxs = [v] * B
                ws = [1.0] * B
            else:
                # deterministic: take first B, or repeat
                if len(nbrs) >= B:
                    idxs = nbrs[:B]
                    ws = w_lists[v][:B] if len(w_lists[v]) >= B else [1.0] * B
                else:
                    rep = (B + len(nbrs) - 1) // len(nbrs)
                    idxs = (nbrs * rep)[:B]
                    ws_src = w_lists[v] if len(w_lists[v]) == len(nbrs) else [1.0] * len(nbrs)
                    ws = (ws_src * rep)[:B]

            neigh_idx[v] = torch.tensor(idxs, device=device, dtype=torch.long)
            neigh_w[v, :, 0] = torch.tensor(ws, device=device, dtype=torch.float32)

        root_feat = x_t                       # [N, F]
        neigh_feat = x_t[neigh_idx]           # [N, B, F]

        # timestamps are zeros in snapshot data
        root_ts = torch.zeros((N,), device=device, dtype=torch.float32)
        neigh_ts = torch.zeros((N, B), device=device, dtype=torch.float32)

        return _StaticBlock(
            root_node_feature=root_feat,
            neighbor_node_feature=neigh_feat,
            neighbor_edge_feature=neigh_w,
            root_ts=root_ts,
            neighbor_ts=neigh_ts,
            n=N,
            b=B,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        H_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block = self._build_block(x_t, edge_index, edge_weight)
        # TransformerAggregator is expected to return [N, out_channels]
        out = self.layer(block)
        return out

