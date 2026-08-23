#SE-HTGNN encoder
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise ImportError("This model requires torch_geometric installed (GCNConv).") from e

class LLM4Init(nn.Module):
    def __init__(
        self,
        reltypes: List[Tuple[str, str, str]],
        llm_feature: Optional[Dict[str, torch.Tensor]] = None,
        llm_dim: int = 4096,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

        # store unique (stype, reltype, dtype)
        self.reltype: List[Tuple[str, str, str]] = []
        for stype, reltype, dtype in reltypes:
            if (stype, reltype, dtype) not in self.reltype:
                self.reltype.append((stype, reltype, dtype))

        self._internal = (llm_feature is None)
        if self._internal:
            ntypes = sorted({s for s, _, _ in self.reltype} | {d for _, _, d in self.reltype})
            # typical LLM_feature is a vector, often 4096x1
            self.llm_feature = nn.ParameterDict({
                ntype: nn.Parameter(torch.randn(llm_dim, 1) * 0.02) for ntype in ntypes
            })
        else:
            self.llm_feature = llm_feature

    def forward(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        feature_dict: Dict[str, torch.Tensor] = {}

        if self._internal:
            for key, param in self.llm_feature.items():
                feature_dict[key] = param
        else:
            for key, feat in self.llm_feature.items():
                feature_dict[key] = feat

        grouped_edges = defaultdict(list)
        for stype, reltype, dtype in self.reltype:
            stype_feat = feature_dict[stype]
            dtype_feat = feature_dict[dtype]
            ip = torch.dot(stype_feat.squeeze(), dtype_feat.squeeze())
            grouped_edges[dtype].append((ip, stype, reltype, dtype))

        normalized_inner_products: Dict[str, torch.Tensor] = {}

        for _dtype, edges in grouped_edges.items():
            inner_products = torch.stack([e[0] for e in edges])  # [R]

            # log requires positive; clamp to eps (structure preserved: log + softmax)
            inner_products = inner_products.clamp(min=self.eps)
            weights = F.softmax(torch.log(inner_products), dim=0)  # [R]

            for i, (_ip, _stype, reltype, _dtype2) in enumerate(edges):
                normalized_inner_products[reltype] = weights[i]

        return normalized_inner_products, feature_dict

class DynamicAtt(nn.Module):
    def __init__(self, d_in: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=1, num_layers=num_layers, batch_first=True)

    def forward(self, x_seq: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        # x_seq: [N, T, D], h0: [layers, N, 1]
        out, _ = self.gru(x_seq, h0)      # [N, T, 1]
        out = out.squeeze(-1)             # [N, T]
        w = out.mean(dim=0)               # [T]
        return F.softmax(w, dim=0)        # [T]


class LinearProj(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.project = nn.Linear(T, 1)

    def forward(self, h_time: torch.Tensor) -> torch.Tensor:
        # h_time: [N, D, T] -> [N, D]
        return self.project(h_time).squeeze(-1)


class SEHTGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,                 # kept for API compatibility
        time_window: int = 12,
        llm_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_window = int(time_window)

        self.dropout = nn.Dropout(dropout)
        self.adapt = nn.Linear(in_channels, out_channels)

        # Intra (spatial) aggregation
        self.gcn = GCNConv(out_channels, out_channels)

        # Homogeneous canonical_etypes equivalent: one node type, one relation
        reltypes = [("node", "edge", "node")]
        self.LLM_init = LLM4Init(reltypes=reltypes, llm_feature=None, llm_dim=llm_dim)

        # Dynamic attention + Linear projection
        self.dynamic_att = DynamicAtt(d_in=out_channels, num_layers=1)
        self.linear_proj = LinearProj(T=self.time_window)

        # Rolling buffer: [N, T, D]
        self._buffer: Optional[torch.Tensor] = None

    def reset_state(self):
        """Call at the start of each independent window (snapshot sequence)."""
        self._buffer = None

    def forward(
        self,
        x_t: torch.Tensor,                    # [N, F]
        edge_index: torch.Tensor,             # [2, E]
        edge_weight: Optional[torch.Tensor] = None,   # [E] or None
        H_prev: Optional[torch.Tensor] = None,        # unused (API compatibility)
    ) -> torch.Tensor:
        device = x_t.device

        # (1) Adaption + dropout
        h = self.dropout(self.adapt(x_t))  # [N, D]

        # (2) Spatial aggregation
        if edge_weight is not None:
            h = self.gcn(h, edge_index, edge_weight)
        else:
            h = self.gcn(h, edge_index)
        h = F.elu(h)  # [N, D]

        # (3) Update rolling time-window buffer (stateful within a window)
        if self._buffer is None or self._buffer.size(0) != h.size(0):
            self._buffer = h.unsqueeze(1).repeat(1, self.time_window, 1)  # [N,T,D]
        else:
            self._buffer = torch.cat([self._buffer[:, 1:, :], h.unsqueeze(1)], dim=1)

        # (4) Repo-structure LLM init -> init_attention dict
        init_attention, _ = self.LLM_init()
        init_scalar = init_attention["edge"].to(device=device)  # scalar (homogeneous)

        # (5) Dynamic attention over the window
        h0 = init_scalar.expand(1, self._buffer.size(0), 1).contiguous()  # [1,N,1]
        w = self.dynamic_att(self._buffer, h0)  # [T]
        fused = torch.sum(self._buffer * w.view(1, -1, 1), dim=1)  # [N, D]

        # (6) Linear projection over the window
        h_time = self._buffer.transpose(1, 2)  # [N, D, T]
        proj = self.linear_proj(h_time)        # [N, D]

        return fused + proj

