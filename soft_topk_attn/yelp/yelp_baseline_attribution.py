#!/usr/bin/env python3
"""
Baseline (DGNN only) for Yelp

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import k_hop_subgraph

from soft_topk_attn.models.metrics_bin_updated import (
    ndcg_at_k_from_logits,
)

from yelp import YelpBipartiteTemporal
from soft_topk_attn.models.integrated_gradients import (
    compute_integrated_gradients,
    compute_node_importance_scores,
)


# =========================
# Timing helpers
# =========================
def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now() -> float:
    return time.perf_counter()


def x_to_batched_sequence(x_flat: torch.Tensor, lags: int) -> torch.Tensor:
    """
    x_flat: [N, lags*F]
    return: [1, lags, N, F]
    """
    if x_flat.dim() != 2:
        raise ValueError(f"x_flat must be 2D, got {tuple(x_flat.shape)}")
    N, D = x_flat.shape
    if D % lags != 0:
        raise ValueError(f"feature dim {D} not divisible by lags={lags}")
    Fdim = D // lags
    x = x_flat.view(N, lags, Fdim)           # [N, lags, F]
    x = x.permute(1, 0, 2).unsqueeze(0)      # [1, lags, N, F]
    return x


def ce_margin_score(logits2: torch.Tensor) -> torch.Tensor:
    # logits for 2-class -> score = logit(1) - logit(0)
    return logits2[..., 1] - logits2[..., 0]


# =========================
# Snapshot container
# =========================
@dataclass
class Snap:
    t: int
    x_flat: torch.Tensor                  # [N, lags*F]
    edge_indices: List[torch.Tensor]      # list length=lags, each [2, E]
    edge_weights: List[torch.Tensor]      # list length=lags, each [E]
    Nb: int                               # number of business nodes (fixed across time in preprocessing)


# =========================
# DGNN backbone
# =========================
class DGNNEncoder(nn.Module):
    """
    Supports per-lag edge_index/edge_weight.
    This is important because Yelp review edges are dynamic.
    """

    def __init__(self, model_name: str, in_channels: int, d_emb: int, K: int, lags: int):
        super().__init__()
        name = str(model_name).strip().upper()
        self.model_name = name
        self.lags = int(lags)
        self.d_emb = d_emb

        if name == "DCRNN":
            from torch_geometric_temporal.nn.recurrent import DCRNN as _DCRNN
            self.cell = _DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)
        elif name == "SEHTGNN":
            from soft_topk_attn.models.SEHTGNN import SEHTGNN as _SEHTGNN
            self.cell = _SEHTGNN(in_channels=in_channels, out_channels=d_emb, K=K, time_window=lags)
        elif name == "TASER":
            from soft_topk_attn.models.taser import TaserTGNNCell as _TASER
            self.cell = _TASER(
                in_channels=in_channels,
                out_channels=d_emb,
                num_neighbors=16,
                dim_time=16,
                att_head=4,
                dropout=0.0,
                time_enc="learnable",
            )
        else:
            raise ValueError(f"Unknown model_name={model_name}")

    def forward(
        self,
        X: torch.Tensor,                          # [1, lags, N, F]
        edge_indices: List[torch.Tensor],         # length=lags
        edge_weights: List[torch.Tensor],         # length=lags
    ) -> torch.Tensor:
        if hasattr(self.cell, "reset_state"):
            self.cell.reset_state()

        if len(edge_indices) != X.size(1) or len(edge_weights) != X.size(1):
            raise ValueError("edge_indices/edge_weights must have length == lags")

        H = None
        for i in range(X.size(1)):
            x_t = X[0, i]  # [N, F]
            ei = edge_indices[i]
            ew = edge_weights[i]
            try:
                H = self.cell(x_t, ei, ew, H)
            except TypeError:
                # some cells take (x, edge_index, edge_weight)
                H = self.cell(x_t, ei, ew)
        return H  # [N, d_emb]
    
    def get_embeddings(
        self,
        X: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """Extract node embeddings (same as forward, but explicit for attribution)"""
        return self.forward(X, edge_indices, edge_weights)


class BinaryCEHead(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.lin(rep)


def pool_subgraph_mean(emb_nodes: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    return emb_nodes[subset].mean(dim=0)


def extract_embeddings_from_encoder(
    encoder: DGNNEncoder,
    x_flat: torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    lags: int,
) -> torch.Tensor:
    """
    Extract node embeddings from the DGNN encoder.
    
    Args:
        encoder: The DGNN encoder model
        x_flat: [N, lags*F] flattened temporal features
        edge_indices: list of edge indices (length=lags)
        edge_weights: list of edge weights (length=lags)
        lags: Number of time lags
        
    Returns:
        embeddings: [N, d_emb] node embeddings
    """
    X = x_to_batched_sequence(x_flat, lags=lags)
    embeddings = encoder.get_embeddings(X, edge_indices, edge_weights)
    return embeddings


def _context_from_subset(
    emb_nodes: torch.Tensor,
    subset: torch.Tensor,
) -> torch.Tensor:
    if subset.numel() == 0:
        return torch.zeros((emb_nodes.size(1),), device=emb_nodes.device)
    return emb_nodes[subset].mean(dim=0)


def plot_deletion_curve(
    fracs: List[float],
    scores: List[float],
    title: str,
    out_path: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.plot(fracs, scores, marker="o")
    plt.xlabel("Fraction of top-k neighbors removed")
    plt.ylabel("NDCG@k")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


@torch.inference_mode()
def compute_sufficiency(
    *,
    head: nn.Module,
    emb_nodes: torch.Tensor,
    emb_a: torch.Tensor,
    subset: torch.Tensor,
    attributions: torch.Tensor,
    k_frac: float,
) -> float:
    """
    Compute sufficiency using Integrated Gradients attributions.
    |f(G) - f(G_Sk)| where G is full subset and G_Sk is top-k nodes by IG importance.
    """
    subset_size = 18
    if subset_size == 0:
        return 0.0

    # k_frac = max(0.0, min(1.0, float(k_frac)))
    k_hard = 18
    
    # Rank nodes by IG importance
    node_scores = compute_node_importance_scores(attributions, aggregation='l2')
    subset_scores = node_scores[subset]
    sorted_indices = torch.argsort(subset_scores, descending=True)
    topk_indices = sorted_indices[:k_hard]
    topk_nodes = subset[topk_indices]

    ctx_full = pool_subgraph_mean(emb_nodes, subset)
    ctx_topk = pool_subgraph_mean(emb_nodes, topk_nodes)

    score_full = torch.sigmoid(ce_margin_score(head(emb_a + ctx_full)))
    score_topk = torch.sigmoid(ce_margin_score(head(emb_a + ctx_topk)))
    return float(abs(score_full - score_topk))


@torch.inference_mode()
def compute_deletion_curve_ndcg(
    *,
    head: nn.Module,
    emb_nodes: torch.Tensor,
    anchor_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    removal_fracs: List[float],
    k_eval: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Deletion curve with NDCG@k using Integrated Gradients attributions.
    anchor_info entries: (emb_a, subset, attributions, y_class)
    where attributions is [N, d_emb] from integrated gradients
    """
    device = emb_nodes.device
    fracs_t = torch.as_tensor(removal_fracs, device=device, dtype=torch.float32)
    if fracs_t.numel() == 0:
        fracs_t = torch.tensor([0.0], device=device)
    if float(fracs_t.min()) > 0.0:
        fracs_t = torch.cat([torch.zeros(1, device=device), fracs_t])
    fracs_t = torch.clamp(fracs_t, 0.0, 1.0)
    fracs_t, _ = torch.sort(fracs_t)

    ndcg_scores = []
    for frac in fracs_t.tolist():
        scores = []
        labels = []
        for emb_a, subset, attributions, y_class in anchor_info:
            # Rank nodes by IG importance (L2 norm of attributions)
            node_scores = compute_node_importance_scores(attributions, aggregation='l2')
            # Get subset node scores
            subset_scores = node_scores[subset]
            sorted_indices = torch.argsort(subset_scores, descending=True)
            
            subset_size = 18  # subset.numel()
            remove_k = int(round(float(frac * subset_size)))
            remove_k = min(max(remove_k, 0), subset_size)

            keep_indices = sorted_indices[remove_k:]
            keep_nodes = subset[keep_indices]

            # if remove_k > 0:
            #     # Keep nodes are those NOT in top-K
            #     keep_indices = sorted_indices[remove_k:]
            #     keep_nodes = subset[keep_indices]
            # else:
            #     keep_nodes = subset

            ctx = pool_subgraph_mean(emb_nodes, keep_nodes)
            rep = emb_a + ctx
            logits2 = head(rep)
            scores.append(ce_margin_score(logits2))
            labels.append(y_class.float())

        scores_t = torch.stack(scores)
        labels_t = torch.stack(labels).squeeze()  # Ensure 1D tensor
        ndcg = ndcg_at_k_from_logits(scores_t, labels_t, k=k_eval)
        ndcg_scores.append(torch.tensor(ndcg, device=device))

    scores_t = torch.stack(ndcg_scores)
    auc = torch.trapz(scores_t, fracs_t).item()
    return fracs_t, scores_t, auc


# =========================
# Task
# =========================
@torch.no_grad()
def _build_u2b_adjacency_for_window(
    ds: YelpBipartiteTemporal,
    t: int,
    window: int,
    Nb: int,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Build:
      - b2users: business -> list of users (unified ids) who reviewed it in [t-window+1 .. t]
      - user_last_biz: user(unified id) -> most recent business(unified id) reviewed in that window
    """
    start = max(0, t - window + 1)

    b2users: Dict[int, List[int]] = {}
    user_last_biz: Dict[int, int] = {}

    # iterate months from old->new to fill last, then overwrite by newer months
    for tt in range(start, t + 1):
        src, dst, _ = ds.get_event_list(tt, direction="u2b", include_edge_attr=False, device=None)
        # src are unified user ids (>=Nb), dst are unified business ids (<Nb)
        if src.numel() == 0:
            continue

        src_list = src.tolist()
        dst_list = dst.tolist()
        for u_uni, b_uni in zip(src_list, dst_list):
            # business->users
            if b_uni not in b2users:
                b2users[b_uni] = [u_uni]
            else:
                b2users[b_uni].append(u_uni)
            # user last business (overwrite as time increases)
            user_last_biz[u_uni] = b_uni

    return b2users, user_last_biz


@torch.no_grad()
def _review_count_at_nb(ds: YelpBipartiteTemporal, t: int, Nb: int) -> torch.Tensor:
    src, dst, _ = ds.get_event_list(t, direction="u2b", include_edge_attr=False, device=None)
    cnt = torch.zeros((Nb,), dtype=torch.long)
    if dst.numel() == 0:
        return cnt
    ones = torch.ones((dst.numel(),), dtype=torch.long)
    cnt.index_add_(0, dst.to(dtype=torch.long), ones)
    return cnt


@torch.no_grad()
def label_peer_quantile_future_count(
    *,
    ds: YelpBipartiteTemporal,
    t_now: int,
    horizon: int,
    anchor_b: int,           # unified business id (0..Nb-1)
    Nb: int,
    recent_window: int,
    quantile_q: float,
    min_reviewers: int,
    min_peers: int,
) -> Optional[int]:
    """
    label = 1 if count_anchor(t+h) > quantile(count_peers(t+h), q)
    """
    t_future = t_now + horizon
    if t_future >= len(ds):
        return None

    b2users, user_last_biz = _build_u2b_adjacency_for_window(ds, t_now, recent_window, Nb)

    users = b2users.get(int(anchor_b), [])
    if len(users) == 0:
        return None
    users = list(set(users))

    if len(users) < int(min_reviewers):
        return None

    peers = []
    for u in users:
        if u in user_last_biz:
            peers.append(user_last_biz[u])

    peers = [b for b in peers if b != int(anchor_b)]
    peers = list(set(peers))

    if len(peers) < int(min_peers):
        return None

    cnt_future = _review_count_at_nb(ds, t_future, Nb=Nb)
    c_anchor = int(cnt_future[int(anchor_b)].item())
    peer_counts = cnt_future[torch.tensor(peers, dtype=torch.long)]
    if peer_counts.numel() == 0:
        return None

    q = float(quantile_q)
    q = max(0.0, min(1.0, q))
    thr = float(torch.quantile(peer_counts.to(dtype=torch.float32), q).item())

    return 1 if float(c_anchor) > thr else 0


# =========================
# Build DGNN snapshots from Yelp
# =========================
@torch.no_grad()
def build_snaps_from_api(
    ds: YelpBipartiteTemporal,
    lags: int,
    device: torch.device,
    bidirectional: bool = True,
    add_type_onehot: bool = True,
) -> Tuple[List[Snap], int, int]:
    """
    Convert ds.to_dynamic_graph_temporal_signal() into Snap list.
    Each Snap contains:
      - x_flat: [N, lags*F]
      - edge_indices/edge_weights: list length=lags (dynamic)
    """
    dgts = ds.to_dynamic_graph_temporal_signal(
        bidirectional=bidirectional,
        add_type_onehot=add_type_onehot,
        device=None,   # keep on CPU; we move per-step to GPU as needed
    )

    d0 = ds.get_hetero(0)
    Nb = int(d0["business"].x.size(0))
    Nu = int(d0["user"].x.size(0))
    N = Nb + Nu

    T = len(ds)
    snaps: List[Snap] = []
    base = lags - 1

    for t in range(base, T):
        feats = []
        eis = []
        ews = []
        for tt in range(t - lags + 1, t + 1):
            g = dgts[tt]  # PyG Data
            feats.append(g.x.cpu())
            eis.append(g.edge_index.cpu())
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                ew = g.edge_attr.view(-1).cpu()
            else:
                ew = torch.ones((g.edge_index.size(1),), dtype=torch.float32)
            ews.append(ew)

        x_lag = torch.stack(feats, dim=1)             # [N, lags, F]
        x_flat = x_lag.reshape(N, -1).contiguous()    # [N, lags*F]

        snaps.append(
            Snap(
                t=int(t),
                x_flat=x_flat,
                edge_indices=eis,
                edge_weights=ews,
                Nb=Nb,
            )
        )

    return snaps, Nb, Nu


# =========================
# Train / eval
# =========================
def _sample_anchors_business_only(Nb: int, k: int, device: torch.device) -> torch.Tensor:
    cand = torch.arange(Nb, device=device)
    if k is None or k < 0 or k >= cand.numel():
        return cand
    perm = torch.randperm(cand.numel(), device=device)
    return cand[perm[:k]]


def _khop_subset(anchor_b: int, edge_index: torch.Tensor, k_hop: int, num_nodes: int) -> torch.Tensor:
    subset, _, _, _ = k_hop_subgraph(
        node_idx=int(anchor_b),
        num_hops=int(k_hop),
        edge_index=edge_index,
        relabel_nodes=False,
        num_nodes=int(num_nodes),
    )
    return subset


def train_step_one_snapshot(
    *,
    args,
    ds: YelpBipartiteTemporal,
    snap: Snap,
    Nb: int,
    encoder: DGNNEncoder,
    head: nn.Module,
    opt: torch.optim.Optimizer,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
) -> Optional[float]:
    encoder.train()
    head.train()

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)  # [N, d_emb]
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_business_only(Nb=Nb, k=args.anchors_train, device=device)

    loss_t = 0.0
    used = 0

    # use last-lag edges for subgraph extraction (CPU)
    ei_now = snap.edge_indices[-1]

    for anchor_b in anchor_indices.tolist():
        subset = _khop_subset(anchor_b, ei_now, args.k_hop, num_nodes=N).to(device=device)

        y = label_peer_quantile_future_count(
            ds=ds,
            t_now=snap.t,
            horizon=args.horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            recent_window=args.recent_window,
            quantile_q=args.quantile,
            min_reviewers=args.min_reviewers,
            min_peers=args.min_peers,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)

        emb_a = emb_nodes[anchor_b]
        ctx = pool_subgraph_mean(emb_nodes, subset)

        # ---- baseline rep ----
        rep = emb_a + ctx
        logits2 = head(rep)

        loss_t = loss_t + ce(logits2.view(1, 2), y_class)
        used += 1

    if used == 0:
        return None

    loss_t = loss_t / float(used)

    opt.zero_grad(set_to_none=True)
    loss_t.backward()
    opt.step()

    return float(loss_t.detach().cpu().item())


def eval_one_snapshot(
    *,
    args,
    ds: YelpBipartiteTemporal,
    snap: Snap,
    Nb: int,
    encoder: DGNNEncoder,
    head: nn.Module,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    horizon: int,
) -> Tuple[float, Dict[str, float]]:
    encoder.eval()
    head.eval()

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_business_only(Nb=Nb, k=args.anchors_eval, device=device)

    anchor_info = []

    loss_sum = 0.0
    used = 0

    ei_now = snap.edge_indices[-1]  # CPU for khop

    for anchor_b in anchor_indices.tolist():
        subset = _khop_subset(anchor_b, ei_now, args.k_hop, num_nodes=N).to(device=device)

        y = label_peer_quantile_future_count(
            ds=ds,
            t_now=snap.t,
            horizon=horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            recent_window=args.recent_window,
            quantile_q=args.quantile,
            min_reviewers=args.min_reviewers,
            min_peers=args.min_peers,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)

        emb_a = emb_nodes[anchor_b]
        ctx = pool_subgraph_mean(emb_nodes, subset)

        rep = emb_a + ctx
        logits2 = head(rep)

        loss_sum += float(ce(logits2.view(1, 2), y_class).item())
        used += 1

        # define classification head for this anchor
        def classification_head(agg_embedding):
            # agg_embedding: [d_emb] -> logits [2]
            # anchor's prediction is based on: emb_a + context where context is the mean of subset embeddings
            logits = head(emb_a + agg_embedding)
            return ce_margin_score(logits)
        
        # Create attention weights that only consider the subset
        attention_weights = torch.zeros(N, device=device)
        if subset.numel() > 0:
            attention_weights[subset] = 1.0 / subset.numel()
        
        # Compute IG attributions
        with torch.enable_grad():
            attributions = compute_integrated_gradients(
                embeddings=emb_nodes,
                classification_head=classification_head,
                baseline_type='zero',
                steps=20,
                target_class=None,
                attention_weights=attention_weights,
            )  # [N, d_emb]
        
        anchor_info.append((emb_a.detach(), subset.detach(), attributions.detach(), y_class.detach()))

    if used == 0:
        return 0.0, {
            "deletion_auc": 0.0,
            "deletion_curve": [],
            "deletion_fracs": [],
            "sufficiency": 0.0,
        }

    avg_loss = loss_sum / float(used)

    metrics = {}

    # Integrated Gradients deletion curve and sufficiency
    if len(anchor_info) > 0:
        fracs_t, del_scores, del_auc = compute_deletion_curve_ndcg(
            head=head,
            emb_nodes=emb_nodes,
            anchor_info=anchor_info,
            removal_fracs=args.deletion_fracs,
            k_eval=args.k_eval,
        )
        metrics.update(
            {
                "deletion_auc": float(del_auc),
                "deletion_curve": del_scores.detach().cpu().tolist(),
                "deletion_fracs": fracs_t.detach().cpu().tolist(),
            }
        )
        
        # IG sufficiency
        suff_vals = []
        for emb_a, subset, attributions, _ in anchor_info:
            suff = compute_sufficiency(
                head=head,
                emb_nodes=emb_nodes,
                emb_a=emb_a,
                subset=subset,
                attributions=attributions,
                k_frac=args.suff_k_frac,
            )
            suff_vals.append(suff)
        metrics["sufficiency"] = float(sum(suff_vals) / max(1, len(suff_vals)))
    else:
        metrics.update({
            "deletion_auc": 0.0,
            "deletion_curve": [],
            "deletion_fracs": [],
            "sufficiency": 0.0,
        })

    if args.save_del_curves and len(anchor_info) > 0:
        # Save IG deletion curve
        out_path = f"yelp_baseline_deletion_curve_ig_h{horizon}.png"
        plot_deletion_curve(
            fracs=metrics["deletion_fracs"],
            scores=metrics["deletion_curve"],
            title=f"Yelp Baseline Deletion Curve (IG) h={horizon}",
            out_path=out_path,
        )
        
        npz_path = f"yelp_baseline_deletion_curve_ig_h{horizon}.npz"
        np.savez(
            npz_path,
            fracs=np.array(metrics["deletion_fracs"]),
            scores=np.array(metrics["deletion_curve"]),
            auc=metrics["deletion_auc"],
        )

    return avg_loss, metrics


def train_one_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ds = YelpBipartiteTemporal(args.pt, map_location="cpu")
    snaps, Nb, Nu = build_snaps_from_api(
        ds,
        lags=args.lags,
        device=device,
        bidirectional=True,
        add_type_onehot=True,
    )
    S = len(snaps)

    F_in = snaps[0].x_flat.size(1) // args.lags

    encoder = DGNNEncoder(
        model_name=args.model,
        in_channels=F_in,
        d_emb=args.d_emb,
        K=args.dcrnn_k,
        lags=args.lags,
    ).to(device)

    head = BinaryCEHead(args.d_emb).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    ce = nn.CrossEntropyLoss()

    W = max(1, min(args.warmup_snaps, S - 1))
    Hs = tuple(args.test_horizons)
    Hmax = max(Hs)

    print(f"\n=== Yelp PeerQuantile Task | BASELINE (DGNN only) | model={args.model} ===")
    print(f"pt={args.pt}")
    print(f"T(raw)={len(ds)}  S(snaps)={S}  Nb={Nb} Nu={Nu} N={Nb+Nu}")
    print(f"lags={args.lags} k-hop={args.k_hop} horizon(train)={args.horizon}")
    print(f"recent_window={args.recent_window} quantile={args.quantile}")
    print(f"evidence: min_reviewers={args.min_reviewers}, min_peers={args.min_peers}")
    print(f"warmup: W={W} epochs={args.epochs_warmup}")
    print(f"rolling horizons={Hs}")

    # ---- warmup ----
    _sync_if_cuda(device)
    t0 = _now()
    for ep in range(args.epochs_warmup):
        losses = []
        for s in range(W):
            loss_val = train_step_one_snapshot(
                args=args,
                ds=ds,
                snap=snaps[s],
                Nb=Nb,
                encoder=encoder,
                head=head,
                opt=opt,
                ce=ce,
                device=device,
            )
            if loss_val is not None:
                losses.append(loss_val)
        avg = sum(losses) / max(1, len(losses))
        print(f"warmup_epoch={ep+1} avg_loss={avg:.6f} (kept={len(losses)})")

    _sync_if_cuda(device)
    warmup_time = _now() - t0

    # ---- rolling ----
    acc = {
        h: {
            "loss": 0.0,
            "deletion_auc": 0.0,
            "sufficiency": 0.0,
            "n": 0,
        }
        for h in Hs
    }

    _sync_if_cuda(device)
    t1 = _now()
    for idx in range(W, S - Hmax):
        train_loss = train_step_one_snapshot(
                        args=args,
                        ds=ds,
                        snap=snaps[idx],
                        Nb=Nb,
                        encoder=encoder,
                        head=head,
                        opt=opt,
                        ce=ce,
                        device=device,
                    )

        print(f"t={idx} | train_loss={train_loss:.6f}" if train_loss is not None else f"t={idx} | train_loss=None")
        
        # Only evaluate on the last timestamp
        if idx == S - Hmax - 1:
            for h in Hs:
                loss_h, m_h = eval_one_snapshot(
                    args=args,
                    ds=ds,
                    snap=snaps[idx],
                    Nb=Nb,
                    encoder=encoder,
                    head=head,
                    ce=ce,
                    device=device,
                    horizon=h,
                )
                acc[h]["loss"] += float(loss_h)
                acc[h]["deletion_auc"] += float(m_h["deletion_auc"])
                acc[h]["sufficiency"] += float(m_h["sufficiency"])
                acc[h]["n"] += 1
            
            msg = [f"t={idx} (final)"]
            for h in Hs:
                n = acc[h]["n"]
                if n > 0:
                    msg.append(
                        f"h{h}:DEL-AUC={acc[h]['deletion_auc'] / n:.3f},"
                        f"SUFF={acc[h]['sufficiency'] / n:.3f}"
                    )
            print("  " + " | ".join(msg))

    _sync_if_cuda(device)
    rolling_time = _now() - t1

    summary = {}
    for h in Hs:
        n = max(1, acc[h]["n"])
        summary[h] = {
            "avg_loss": acc[h]["loss"] / n,
            "deletion_auc": acc[h]["deletion_auc"] / n,
            "sufficiency": acc[h]["sufficiency"] / n,
            "steps": acc[h]["n"],
        }

    return warmup_time, rolling_time, summary


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--pt", type=str, required=True, help="preprocessed Yelp .pt (from yelp_process.py)")
    p.add_argument("--model", type=str, default="TASER", choices=["DCRNN", "SEHTGNN", "TASER"])

    p.add_argument("--cpu", action="store_true")

    # temporal
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--test_horizons", type=int, nargs="+", default=[1])

    # task params
    p.add_argument("--recent_window", type=int, default=6, help="how many recent months to define reviewers/peers")
    p.add_argument("--quantile", type=float, default=0.5, help="0.5=median, 0.75=top-quartile threshold")
    p.add_argument("--min_reviewers", type=int, default=5)
    p.add_argument("--min_peers", type=int, default=5)

    # subgraph
    p.add_argument("--k_hop", type=int, default=2)

    # model dims
    p.add_argument("--d_emb", type=int, default=32)
    p.add_argument("--dcrnn_k", type=int, default=2)

    # opt
    p.add_argument("--lr", type=float, default=1e-3)

    # sampling
    p.add_argument("--anchors_train", type=int, default=256)
    p.add_argument("--anchors_eval", type=int, default=512)
    p.add_argument("--k_eval", type=int, default=50)
    p.add_argument("--deletion_fracs", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--suff_k_frac", type=float, default=1)
    p.add_argument("--save_del_curves", action="store_true", default=True)

    # warmup/rolling
    p.add_argument("--warmup_snaps", type=int, default=6)
    p.add_argument("--epochs_warmup", type=int, default=3)
    #p.add_argument("--anneal", type=float, default=0.9)        # kept for CLI compatibility (unused here)
    p.add_argument("--print_every", type=int, default=1)

    # # attention params (kept for CLI compatibility)
    # p.add_argument("--tau", type=float, default=0.6)
    # p.add_argument("--init_k_frac", type=float, default=0.05)
    # p.add_argument("--k_min", type=float, default=0.01)
    # p.add_argument("--k_max", type=float, default=0.2)
    # p.add_argument("--k_abs_min", type=int, default=10)
    # p.add_argument("--k_abs_max", type=int, default=50)
    #
    # # loss weights (kept for CLI compatibility)
    # p.add_argument("--beta_div", type=float, default=0.0)
    # p.add_argument("--beta_metric", type=float, default=0.5)
    # p.add_argument("--beta_mixup", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    warm0, roll0, summ0 = train_one_run(args)

    print("\n===== Summary =====")
    print(f"Baseline (IG): warmup_time={warm0:.2f}s rolling_time={roll0:.2f}s")
    for h, m in summ0.items():
        print(
            f"  horizon={h} avg_loss={m['avg_loss']:.6f} "
            f"DEL-AUC={m['deletion_auc']:.3f} "
            f"SUFF={m['sufficiency']:.3f} "
            f"steps={m['steps']}"
        )


if __name__ == "__main__":
    main()
