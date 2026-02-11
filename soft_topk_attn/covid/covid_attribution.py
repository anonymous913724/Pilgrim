#!/usr/bin/env python3
"""
COVID county attribution code

Task (COVID):
For an anchor county v at time t, define subgraph S(v) as k-hop neighborhood on the static county adjacency graph.
Define subgraph score at time t:
  score_t(v) = mean( x_t[u] for u in S(v) ) / sqrt(|S(v)|)      (density-normalized mean)
Label:
  y_t(v) = 1 if score_{t+i}(v) > tau_thr else 0
where tau_thr is computed from TRAIN only using percentile q_thr (default 0.85).

Attribution:
Run soft-topk attention with dynamic k on the full graph
Compute sufficiency and deletion curves on the top-k induced subgraph, where k is estimated by the attention module.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import time
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# --- FAISS candidate retriever ---
try:
    from soft_topk_attn.models.faiss_gpu_retriever import FaissGpuRetriever
except Exception:
    FaissGpuRetriever = None  # type: ignore

from torch_geometric.utils import k_hop_subgraph

# ---- project modules ----
from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.diversity_loss_f import diversity_loss
from soft_topk_attn.models.mixup_cap import MixupWithMemory
from soft_topk_attn.models.metric_loss import MetricLoss
from soft_topk_attn.models.metrics_bin_updated import (
    f1_from_logits,
)

# ---- COVID API ----
from soft_topk_attn.data.covid import build_nyt_covid_static_graph_temporal_signal


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


@torch.inference_mode()
def compute_sufficiency(
    attention: QKOnlySoftTopKAttention,
    head: nn.Module,
    emb_q: torch.Tensor,
    emb_a: torch.Tensor,
    emb_nodes: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    attn: torch.Tensor,
    soft_mask: torch.Tensor,
    k_estimate: torch.Tensor,
    tau: float,
) -> float:
    """
    Sufficiency: |f(G) - f(G_Sk)|, where G_Sk keeps only top-k nodes by attention.
    Smaller is better.
    """
    device = emb_nodes.device
    base_mask = torch.ones(emb_nodes.size(0), dtype=torch.bool, device=device) if node_mask is None else node_mask.bool()

    # Full graph prediction
    context_full, _, _, _, _, _, _, _ = attention(
        emb_q=emb_q,
        emb_a=emb_a,
        emb_nodes=emb_nodes,
        node_mask=node_mask,
        tau=tau,
        return_intermediates=True,
    )
    logits_full = head(emb_a + emb_q + context_full)
    score_full = ce_margin_score(logits_full)
    score_full = torch.sigmoid(score_full)

    # Rank nodes by importance (soft mask x attention)
    importance = (soft_mask * attn) * base_mask.to(dtype=soft_mask.dtype)
    order = torch.argsort(importance, descending=True)

    valid_k = int(base_mask.sum().item())
    k_hard = max(1, min(int(round(float(k_estimate.detach().item()))), valid_k))

    induced_mask = torch.zeros_like(base_mask)
    induced_mask[order[:k_hard]] = True
    induced_mask = induced_mask & base_mask

    context_induced, _, _, _, _, _, _, _ = attention(
        emb_q=emb_q,
        emb_a=emb_a,
        emb_nodes=emb_nodes,
        node_mask=induced_mask,
        tau=tau,
        return_intermediates=True,
    )
    logits_induced = head(emb_a + emb_q + context_induced)
    score_induced = ce_margin_score(logits_induced)
    score_induced = torch.sigmoid(score_induced)

    return abs(score_full - score_induced)


@torch.inference_mode()
def compute_deletion_curve(
    encoder: 'DGNNEncoder',
    head: nn.Module,
    labels_batch: torch.Tensor,
    emb_q: torch.Tensor,
    anchor_idx: List[int],
    emb_nodes_full: Optional[torch.Tensor],
    X: torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    attn_batch: List[torch.Tensor],
    soft_mask_batch: List[torch.Tensor],
    k_batch: List[torch.Tensor],
    removal_fracs: List[float],
    k_eval: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Deletion curve on COVID: remove top-k important nodes and recompute embeddings.
    Uses F1 as the metric instead of NDCG.
    Returns (fractions, f1_scores, auc).
    """

    device = X.device

    if labels_batch.dim() != 1:
        labels_batch = labels_batch.squeeze()

    base_mask = torch.ones(X.size(2), dtype=torch.bool, device=device)
    valid_k = int(base_mask.sum().item())

    fracs_t = torch.as_tensor(removal_fracs, device=device, dtype=torch.float)
    if fracs_t.numel() == 0:
        fracs_t = torch.tensor([0.0], device=device)
    if float(fracs_t.min()) > 0.0:
        fracs_t = torch.cat([torch.zeros(1, device=device), fracs_t])
    fracs_t = torch.clamp(fracs_t, 0.0, 1.0)
    fracs_t, _ = torch.sort(fracs_t)

    order = []
    for i in range(len(soft_mask_batch)):
        attn = attn_batch[i]
        soft_mask = soft_mask_batch[i]
        importance = (soft_mask * attn) * base_mask.to(dtype=soft_mask.dtype)
        order.append(torch.argsort(importance, descending=True))

    f1_scores = []

    for frac in fracs_t.tolist():
        all_scores = []
        for i in range(len(anchor_idx)):
            k_estimate = k_batch[i]
            k_hard = max(1, min(int(round(float(k_estimate.detach().item()))), valid_k))
            remove_k = int(round(float(frac * k_hard)))
            if i == 0:  # Print debug info for first anchor only
                print(f"    Deletion frac={frac:.2f}, k_hard={k_hard}, remove_k={remove_k}, valid_k={valid_k}")
            remove_k = min(max(remove_k, 0), valid_k)

            if remove_k == 0 and emb_nodes_full is not None:
                mask = base_mask
                emb_nodes_del = emb_nodes_full
            else:
                mask = base_mask.clone()
                if remove_k > 0:
                    mask[order[i][:remove_k]] = False

                pruned_eis = []
                pruned_ews = []
                for ei, ew in zip(edge_indices, edge_weights):
                    edge_src = ei[0]
                    edge_dst = ei[1]
                    edge_keep = mask[edge_src] & mask[edge_dst]
                    pruned_eis.append(ei[:, edge_keep])
                    pruned_ews.append(ew[edge_keep] if ew is not None else None)

                mask_f = mask.to(dtype=X.dtype).view(1, 1, -1, 1)
                X_masked = X * mask_f

                emb_nodes_del = encoder(X_masked, pruned_eis, pruned_ews)

            emb_a = emb_nodes_del[anchor_idx[i]]
            
            attn = attn_batch[i]
            soft_mask = soft_mask_batch[i]
            mask_f = mask.to(dtype=soft_mask.dtype)
            w = (soft_mask * attn * mask_f).unsqueeze(-1)

            context_del = torch.sum(w * emb_nodes_del, dim=0)

            rep = context_del
            logits = head(rep)
            score = ce_margin_score(logits)
            all_scores.append(score)

        all_scores_t = torch.stack(all_scores)
        f1 = f1_from_logits(all_scores_t, labels_batch)
        f1_scores.append(f1)

    scores_t = torch.tensor(f1_scores, device=device)
    auc = torch.trapz(scores_t, fracs_t).item()
    return fracs_t, scores_t, auc


# =========================
# Snapshot container 
# =========================
@dataclass
class Snap:
    t: int
    x_flat: torch.Tensor                  # [N, lags*F]
    edge_indices: List[torch.Tensor]      # list length=lags, each [2, E]
    edge_weights: List[torch.Tensor]      # list length=lags, each [E]
    Nb: int                               # keep name "Nb"; for COVID set Nb=N (all counties)


# =========================
# DGNN backbone 
# =========================
class DGNNEncoder(nn.Module):
    """
    Supports per-lag edge_index/edge_weight.
    (For COVID, edges are static; we still pass per-lag copies to keep the same interface.)
    """

    def __init__(self, model_name: str, in_channels: int, d_emb: int, K: int, lags: int):
        super().__init__()
        name = str(model_name).strip().upper()
        self.model_name = name
        self.lags = int(lags)

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


class BinaryCEHead(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.lin(rep)


# =========================
# COVID: build DGNN snapshots
# =========================
@torch.no_grad()
def build_snaps_from_covid_api(
    dataset,   # StaticGraphTemporalSignal
    meta,      # DatasetMeta
    lags: int,
) -> Tuple[List[Snap], int]:
    """
    Convert StaticGraphTemporalSignal into Snap list
    - Static edges: same edge_index for all lags
    - Features: dataset.features[t] is np.ndarray [N, F]
    build x_flat as concat over lags: [N, lags*F]
    """
    # edge_index in covid.py is numpy int64 with shape [2, E_dir]
    edge_index_np = dataset.edge_index
    if isinstance(edge_index_np, np.ndarray):
        edge_index_cpu = torch.from_numpy(edge_index_np).to(dtype=torch.long)
    else:
        # just in case
        edge_index_cpu = torch.as_tensor(edge_index_np, dtype=torch.long)

    N = int(dataset.features[0].shape[0])
    F = int(dataset.features[0].shape[1])
    T = len(dataset.features)

    snaps: List[Snap] = []
    base = lags - 1

    for t in range(base, T):
        feats = []
        eis = []
        ews = []
        for tt in range(t - lags + 1, t + 1):
            x_np = dataset.features[tt]
            x_cpu = torch.from_numpy(x_np).to(dtype=torch.float32)  # [N, F]
            feats.append(x_cpu)

            eis.append(edge_index_cpu)  # CPU
            ews.append(torch.ones((edge_index_cpu.size(1),), dtype=torch.float32))  # CPU

        x_lag = torch.stack(feats, dim=1)          # [N, lags, F]
        x_flat = x_lag.reshape(N, -1).contiguous() # [N, lags*F]

        snaps.append(
            Snap(
                t=int(t),
                x_flat=x_flat,
                edge_indices=eis,
                edge_weights=ews,
                Nb=N,  # keep name "Nb" for minimal changes; for COVID Nb == N counties
            )
        )

    return snaps, N


# =========================
# COVID task: k-hop subset cache + score/label
# =========================
def _khop_subset_cached(
    anchor: int,
    edge_index_cpu: torch.Tensor,
    k_hop: int,
    num_nodes: int,
    cache: Dict[int, torch.Tensor],
) -> torch.Tensor:
    if int(anchor) in cache:
        return cache[int(anchor)]
    subset, _, _, _ = k_hop_subgraph(
        node_idx=int(anchor),
        num_hops=int(k_hop),
        edge_index=edge_index_cpu,
        relabel_nodes=False,
        num_nodes=int(num_nodes),
    )
    cache[int(anchor)] = subset
    return subset


@torch.no_grad()
def _subgraph_score_density_normalized(
    x_t: torch.Tensor,            # [N] (node signal at some time)
    subset_cpu: torch.Tensor,     # [|S|] on CPU
) -> float:
    # density-normalized mean: mean / sqrt(|S|)
    if subset_cpu.numel() <= 0:
        return float("-inf")
    vals = x_t[subset_cpu.to(dtype=torch.long)]
    mean_val = float(vals.mean().item())
    denom = math.sqrt(float(subset_cpu.numel()))
    return mean_val / max(1e-12, denom)


@torch.no_grad()
def _compute_tau_thr_from_train(
    *,
    snaps: List[Snap],
    anchors_cpu: torch.Tensor,        # [A] anchors to consider (filtered)
    k_hop: int,
    horizon: int,
    q_thr: float,
    train_last_t: int,          # last t (in snap index space) included in train INPUT
) -> float:
    """
    tau_thr computed on TRAIN only:
      use t in [0 .. train_last_t] but label uses t+horizon
      so valid t are [0 .. train_last_t - horizon]
    score is computed at future time t+horizon.
    """
    q = float(q_thr)
    q = max(0.0, min(1.0, q))

    if train_last_t - horizon < 0:
        raise ValueError("train_last_t too small for given horizon")

    # static edge index (cpu) from any snap lag
    edge_index_cpu = snaps[0].edge_indices[-1]
    N = int(snaps[0].Nb)

    cache: Dict[int, torch.Tensor] = {}
    scores = []

    # node signal is feature[:,0] at each snapshot (already in x_flat)
    for t in range(0, int(train_last_t - horizon) + 1):
        t_f = t + int(horizon)
        # take current node signal at future snapshot
        x_f_flat = snaps[t_f].x_flat  # CPU [N, lags*F]
        # for covid we use feature[:,0] of the LAST lag in that snap’s x_flat.
        # in build, for each snap, x_flat is concat over lags, and F can be >=1.
        # index: last lag offset = (lags-1)*F, then +0
        # for COVID always use lags=1 in practice;
        # assume node signal is the first feature of x_flat.
        x_sig = x_f_flat[:, 0].to(dtype=torch.float32)  # [N]

        for a in anchors_cpu.tolist():
            subset = _khop_subset_cached(a, edge_index_cpu, k_hop, num_nodes=N, cache=cache)
            sc = _subgraph_score_density_normalized(x_sig, subset)
            if math.isfinite(sc):
                scores.append(sc)

    if len(scores) == 0:
        raise RuntimeError("No scores collected for tau_thr computation.")

    scores_t = torch.tensor(scores, dtype=torch.float32)
    tau_thr = float(torch.quantile(scores_t, q).item())
    return tau_thr


@torch.no_grad()
def label_peer_quantile_future_count(
    *,
    snaps: List[Snap],
    t_now: int,
    horizon: int,
    anchor_b: int,            # county id
    Nb: int,                  # Nb==N counties
    k_hop: int,
    tau_thr: float,           # label threshold (NOT attention tau)
    subset_cache: Dict[int, torch.Tensor],
) -> Optional[int]:
    """
      y = 1 if score_{t+h}(anchor) > tau_thr else 0
    score is density-normalized mean over k-hop subset on static graph, using node signal feature[:,0].
    """
    t_f = int(t_now) + int(horizon)
    if t_f >= len(snaps):
        return None

    edge_index_cpu = snaps[0].edge_indices[-1]
    x_f_flat = snaps[t_f].x_flat  # CPU
    x_sig = x_f_flat[:, 0].to(dtype=torch.float32)  # [N]

    subset = _khop_subset_cached(anchor_b, edge_index_cpu, k_hop, num_nodes=Nb, cache=subset_cache)
    if subset.numel() <= 0:
        return None

    sc = _subgraph_score_density_normalized(x_sig, subset)
    if not math.isfinite(sc):
        return None

    return 1 if float(sc) > float(tau_thr) else 0


# =========================
# Split helpers
# =========================
def _find_last_t_with_start_leq(meta, date_str: str) -> int:
    target = torch.tensor(int(torch.tensor(0)))  # dummy to keep style stable
    # meta.snapshot_starts is List[pd.Timestamp]
    import pandas as pd
    d = pd.to_datetime(date_str)
    last = -1
    for i, s in enumerate(meta.snapshot_starts):
        if s <= d:
            last = i
    return int(last)


def _t_to_snap_idx(t_raw: int, lags: int) -> int:
    # snaps index starts at base=(lags-1) relative to raw snapshot index
    # snap.t stores raw snapshot index
    # In snap build: snap.t == raw t
    return int(t_raw)


# =========================
# Train / eval
# =========================
def train_one_run(args, use_faiss: bool):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset, meta = build_nyt_covid_static_graph_temporal_signal(
        nyt_us_counties_csv=args.nyt_csv,
        census_adjacency_txt=args.adj_txt,
        date_start=args.date_start,
        date_end=args.date_end,
        snapshot_days=args.snapshot_days,
        feature_mode=args.feature_mode,
    )

    snaps, N = build_snaps_from_covid_api(
        dataset,
        meta,
        lags=args.lags,
    )
    S = len(snaps)

    # feature dims
    F_in = snaps[0].x_flat.size(1) // args.lags

    encoder = DGNNEncoder(
        model_name=args.model,
        in_channels=F_in,
        d_emb=args.d_emb,
        K=args.dcrnn_k,
        lags=args.lags,
    ).to(device)

    head = BinaryCEHead(args.d_emb).to(device)

    # attention + mixup + metric
    query_embed = nn.Embedding(2, args.d_emb).to(device)
    attention = QKOnlySoftTopKAttention(
        d_in=args.d_emb,
        d_out=args.d_emb,
        tau=args.tau,
        init_k_frac=args.init_k_frac,
        k_min=args.k_min,
        k_max=args.k_max,
        normalize_qk=False,
        newton_iters=15,
        newton_damping=1.0,
        k_abs_min=args.k_abs_min,
        k_abs_max=args.k_abs_max,
    ).to(device)

    # -------------------------
    # FAISS retriever (optional)
    # -------------------------
    retriever = None
    if use_faiss:
        if FaissGpuRetriever is None:
            raise RuntimeError('use_faiss=True but faiss_gpu_retriever.py is not importable')
        retriever = FaissGpuRetriever(device=device, metric=args.faiss_metric)

    metric_loss = MetricLoss(num_classes=2, d_emb=args.d_emb, code_size=args.d_emb, device=device).to(device)
    mixup = MixupWithMemory(num_classes=2, d_emb=args.d_emb, device=device)

    params = list(encoder.parameters()) + list(head.parameters()) + list(query_embed.parameters()) + list(attention.parameters()) + list(metric_loss.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    ce = nn.CrossEntropyLoss()

    # -------------------------
    # Anchor filtering (degree==0 removed) - CPU once
    # -------------------------
    edge_index_cpu = snaps[0].edge_indices[-1]
    deg = torch.zeros((N,), dtype=torch.long)
    deg.index_add_(0, edge_index_cpu[0], torch.ones((edge_index_cpu.size(1),), dtype=torch.long))
    keep = (deg > 0)
    anchors_all_cpu = torch.nonzero(keep, as_tuple=False).view(-1).to(dtype=torch.long)

    # -------------------------
    # split (train/test only)
    # -------------------------
    # meta.snapshot_starts are raw snapshot windows (length = len(dataset.features))
    # snaps are built from raw index base=(lags-1), but snap.t == raw index.
    train_last_raw = _find_last_t_with_start_leq(meta, args.train_end_start)
    test_first_raw = train_last_raw + 1

    # internal holdout: last H snapshots of train (in raw index space)
    holdout_len = int(args.holdout_snaps)
    if holdout_len < 0:
        holdout_len = 0

    # valid raw t range for snaps we actually have: base..(T-1)
    base_raw = snaps[0].t
    last_raw = snaps[-1].t

    # clamp split points into available range
    train_last_raw = min(int(train_last_raw), int(last_raw))
    test_first_raw = min(int(test_first_raw), int(last_raw) + 1)

    # train raw indices: base_raw..train_last_raw
    # holdout raw indices: (train_last_raw-holdout_len+1)..train_last_raw
    holdout_start_raw = max(int(base_raw), int(train_last_raw - holdout_len + 1))
    train_inner_last_raw = max(int(base_raw) - 1, int(holdout_start_raw - 1))

    # convert to snap list indices by selecting snaps whose snap.t in range
    train_inner_ids = [i for i, sp in enumerate(snaps) if int(sp.t) <= int(train_inner_last_raw)]
    holdout_ids = [i for i, sp in enumerate(snaps) if int(holdout_start_raw) <= int(sp.t) <= int(train_last_raw)]
    test_ids = [i for i, sp in enumerate(snaps) if int(sp.t) >= int(test_first_raw)]
    #print(f"Train ids: {train_inner_ids} \n Test ids: {test_ids}")

    # need labels at t+horizon, so exclude tail that doesn't have future
    def _filter_valid(ids: List[int]) -> List[int]:
        out = []
        for i in ids:
            t_raw = int(snaps[i].t)
            if (t_raw + int(args.horizon)) <= int(last_raw):
                out.append(i)
        return out

    train_inner_ids = _filter_valid(train_inner_ids)
    holdout_ids = _filter_valid(holdout_ids)
    test_ids = _filter_valid(test_ids)

    # -------------------------
    # tau_thr (LABEL THRESHOLD) computed on train ONLY
    # -------------------------
    train_ids = sorted(list(set(train_inner_ids + holdout_ids)))
    if len(train_ids) == 0:
        raise RuntimeError("train_ids is empty after filtering.")

    # train_last_t in snap-index space: last raw t among train snaps
    train_last_raw_effective = max(int(snaps[i].t) for i in train_ids)

    tau_thr = _compute_tau_thr_from_train(
        snaps=snaps,
        anchors_cpu=anchors_all_cpu,
        k_hop=args.k_hop,
        horizon=args.horizon,
        q_thr=args.q_thr,
        train_last_t=train_last_raw_effective,
    )

    print(f"\n=== COVID Hotspot Exceedance Task | model={args.model} | FAISS={'ON' if use_faiss else 'OFF'} ===")
    print(f"nyt_csv={args.nyt_csv}")
    print(f"adj_txt={args.adj_txt}")
    print(f"T(raw)={len(dataset.features)}  S(snaps)={S}  N(counties)={N}")
    print(f"lags={args.lags} k-hop={args.k_hop} horizon(train)={args.horizon}")
    print(f"train_end_start={args.train_end_start}  holdout_snaps={args.holdout_snaps}")
    print(f"anchors kept (deg>0)={int(anchors_all_cpu.numel())}/{N}")
    print(f"tau_thr(label) percentile={args.q_thr:.2f} computed on train = {tau_thr:.6f}")
    print(f"(NOTE) args.tau is ATTENTION temperature; tau_thr is LABEL threshold.")

    # ---- warmup ----
    W = max(1, min(args.warmup_snaps, len(train_inner_ids) - 1 if len(train_inner_ids) > 1 else 1))
    print(f"warmup: W={W} epochs={args.epochs_warmup}")

    _sync_if_cuda(device)
    t0 = _now()
    for ep in range(args.epochs_warmup):
        tau_ep = args.tau * (args.anneal ** ep)
        beta_metric_ep = args.beta_metric * (args.anneal ** ep)
        beta_mixup_ep = args.beta_mixup * (args.anneal ** ep)

        # FAISS warmup policy
        use_faiss_ep = bool(use_faiss) and (ep == int(args.epochs_warmup) - 1)

        losses = []
        subset_cache_cpu: Dict[int, torch.Tensor] = {}  # task subgraph cache (CPU)
        for k in range(min(W, len(train_inner_ids))):
            idx = train_inner_ids[k]
            loss_val = train_step_one_snapshot(
                args=args,
                snaps=snaps,
                snap=snaps[idx],
                Nb=N,
                anchors_all_cpu=anchors_all_cpu,
                encoder=encoder,
                head=head,
                attention=attention,
                query_embed=query_embed,
                metric_loss=metric_loss,
                mixup=mixup,
                opt=opt,
                ce=ce,
                device=device,
                tau=tau_ep,
                beta_metric=beta_metric_ep,
                beta_mixup=beta_mixup_ep,
                use_faiss=use_faiss_ep,
                retriever=retriever,
                faiss_use_topm_init=bool(use_faiss_ep),
                tau_thr=tau_thr,
                subset_cache_cpu=subset_cache_cpu,
            )
            if loss_val is not None:
                losses.append(loss_val)
        avg = sum(losses) / max(1, len(losses))
        print(f"warmup_epoch={ep+1} avg_loss={avg:.6f} tau(attn)={tau_ep:.4f} (kept={len(losses)})")

    _sync_if_cuda(device)
    warmup_time = _now() - t0

    def _rolling_history_ids(all_ids_sorted: List[int], cur_idx: int, history: int) -> List[int]:
        # all_ids_sorted: sorted snapshot indices (in snaps list index space)
        # cur_idx: current snapshot index to evaluate
        # history: 0 disabled, -1 expanding, H>=1 sliding last H
        prev_ids = [x for x in all_ids_sorted if x < cur_idx]
        if int(history) < 0:
            return prev_ids
        return prev_ids[-int(history):]

    # ---- rolling / prequential mode (always on) ----
    print("\n=== Rolling / Prequential (train -> Holdout -> Test) ===")
    _sync_if_cuda(device)
    t1 = _now()

    # evaluate chronologically across (train + test)
    all_ids_sorted = sorted(list(set(train_inner_ids + holdout_ids + test_ids)))
    train_inner_set = set(train_inner_ids)
    holdout_set = set(holdout_ids)
    test_set = set(test_ids)
    #print(f"Train snap ids: {train_inner_set} \t Test snap ids: {test_set}")

    acc_train = {"loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "n": 0}
    acc_hold  = {"loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "n": 0}
    acc_test  = {"loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "n": 0}

    subset_cache_cpu: Dict[int, torch.Tensor] = {}
    last_del_curve_data = None

    for ii, idx in enumerate(all_ids_sorted):
        # rolling history window from the past
        hist_ids = _rolling_history_ids(all_ids_sorted, cur_idx=idx, history=int(args.roll_history))

        # 1) train on history window (if any)
        if len(hist_ids) > 0:
            encoder.train()
            head.train()
            attention.train()
            query_embed.train()

            for ep in range(int(args.roll_epochs)):
                for jj, jdx in enumerate(hist_ids):
                    tau_j = args.tau * (args.anneal ** (jj / max(1, len(hist_ids))))
                    beta_metric_j = args.beta_metric * (args.anneal ** (jj / max(1, len(hist_ids))))
                    beta_mixup_j = args.beta_mixup * (args.anneal ** (jj / max(1, len(hist_ids))))

                    _ = train_step_one_snapshot(
                        args=args,
                        snaps=snaps,
                        snap=snaps[jdx],
                        Nb=N,
                        anchors_all_cpu=anchors_all_cpu,
                        encoder=encoder,
                        head=head,
                        attention=attention,
                        query_embed=query_embed,
                        metric_loss=metric_loss,
                        mixup=mixup,
                        opt=opt,
                        ce=ce,
                        device=device,
                        tau=tau_j,
                        beta_metric=beta_metric_j,
                        beta_mixup=beta_mixup_j,
                        use_faiss=use_faiss,
                        retriever=retriever,
                        faiss_use_topm_init=False,
                        tau_thr=tau_thr,
                        subset_cache_cpu=subset_cache_cpu,
                    )

        # 2) (optional) burn-in: skip metrics for first B steps
        if ii < int(args.burnin_eval):
            continue

        # 2) eval current idx (only last snapshot)
        if idx != all_ids_sorted[-1]:
            continue

        tau_i = args.tau * (args.anneal ** (ii / max(1, len(all_ids_sorted))))
        loss_h, m_h = eval_one_snapshot(
            args=args,
            snaps=snaps,
            snap=snaps[idx],
            Nb=N,
            anchors_all_cpu=anchors_all_cpu,
            encoder=encoder,
            head=head,
            attention=attention,
            query_embed=query_embed,
            ce=ce,
            device=device,
            tau=tau_i,
            horizon=args.horizon,
            use_faiss=bool(use_faiss),
            retriever=retriever,
            tau_thr=tau_thr,
            subset_cache_cpu=subset_cache_cpu,
        )

        if idx in train_inner_set:
            acc = acc_train
            name = "train"
        elif idx in holdout_set:
            acc = acc_hold
            name = "holdout"
        else:
            acc = acc_test
            name = "test"

        acc["loss"] += float(loss_h)
        acc["deletion_auc"] += float(m_h["deletion_auc"])
        acc["sufficiency"] += float(m_h["sufficiency"])
        acc["n"] += 1

        last_del_curve_data = {
            "fracs": m_h["deletion_fracs"],
            "scores": m_h["deletion_curve"],
            "auc": m_h["deletion_auc"],
        }

        if (ii % max(1, args.print_every)) == 0 and name == "train":
            print(
                f"  t={snaps[idx].t:<3d}  loss={float(loss_h):.4f}  "
                f"deletion_auc={float(m_h['deletion_auc']):.4f}  sufficiency={float(m_h['sufficiency']):.4f}"
            )

        break

    if last_del_curve_data is not None:
        fracs = last_del_curve_data["fracs"]
        scores = last_del_curve_data["scores"]
        if len(fracs) > 0 and len(scores) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(fracs, scores, marker="o", linewidth=2)
            plt.xlabel("Fraction of top-k nodes deleted")
            plt.ylabel("F1")
            plt.title(f"COVID Deletion Curve (Horizon={args.horizon})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = f"covid_deletion_curve_h{args.horizon}.png"
            plt.savefig(out_path, dpi=150)
            print(f"Saved deletion curve plot to {out_path}")
            plt.close()

            np.savez(
                f"covid_deletion_curve_h{args.horizon}.npz",
                fracs=np.array(fracs),
                scores=np.array(scores),
                auc=last_del_curve_data["auc"],
            )
            print(f"Saved deletion curve data to covid_deletion_curve_h{args.horizon}.npz")

    def _finalize(acc: Dict[str, float]) -> Dict[str, float]:
        n = int(acc["n"])
        if n <= 0:
            return {"avg_loss": 0.0, "deletion_auc": 0.5, "sufficiency": 0.0, "steps": 0}
        return {
            "avg_loss": acc["loss"] / n,
            "deletion_auc": acc["deletion_auc"] / n,
            "sufficiency": acc["sufficiency"] / n,
            "steps": n,
        }

    res_train = _finalize(acc_train)
    res_hold = _finalize(acc_hold)
    res_test = _finalize(acc_test)

    _sync_if_cuda(device)
    run_time = _now() - t1
    return warmup_time, run_time, {"train": res_train, "holdout": res_hold, "test": res_test}




def _get_h_idx(h: int) -> int:
    return 0 if int(h) == 1 else 1


def _sample_anchors_business_only(Nb: int, k: int, device: torch.device) -> torch.Tensor:
    cand = torch.arange(Nb, device=device)
    if k is None or k < 0 or k >= cand.numel():
        return cand
    perm = torch.randperm(cand.numel(), device=device)
    return cand[perm[:k]]


def _sample_anchors_from_filtered_cpu(anchors_all_cpu: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    # sample by randperm
    a = anchors_all_cpu.to(device=device)
    if k is None or k < 0 or k >= a.numel():
        return a
    perm = torch.randperm(a.numel(), device=device)
    return a[perm[:k]]


# =========================
# Train step
# =========================
def train_step_one_snapshot(
    *,
    args,
    snaps: List[Snap],
    snap: Snap,
    Nb: int,
    anchors_all_cpu: torch.Tensor,
    encoder: DGNNEncoder,
    head: nn.Module,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    metric_loss: MetricLoss,
    mixup: MixupWithMemory,
    opt: torch.optim.Optimizer,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau: float,                   # attention temperature
    beta_metric: float,
    beta_mixup: float,
    use_faiss: bool,
    retriever: Optional[object],
    faiss_use_topm_init: bool = False,
    tau_thr: float = 0.0,         # LABEL threshold
    subset_cache_cpu: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[float]:
    encoder.train()
    head.train()
    attention.train()
    query_embed.train()
    metric_loss.train()

    if subset_cache_cpu is None:
        subset_cache_cpu = {}

    # build X
    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    # per-lag edges to device
    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)  # [N, d_emb]
    N = emb_nodes.size(0)

    # NOTE: use filtered anchors for COVID (degree>0)
    anchor_indices = _sample_anchors_from_filtered_cpu(anchors_all_cpu, k=args.anchors_train, device=device)

    # =========================
    # FAISS (optional): build index once per snapshot and precompute candidates for all anchors
    # =========================
    cand_batch = None
    if use_faiss:
        if retriever is None:
            raise RuntimeError('use_faiss=True but retriever is None')
        valid_idx = torch.arange(N, device=device, dtype=torch.long)
        K_all = attention.W_k(emb_nodes).detach()
        retriever.build(
            K_all,
            valid_idx=valid_idx,
            normalize=getattr(attention, 'normalize_qk', False),
            require_torch_gpu=bool(args.faiss_require_torch_gpu),
        )
        if faiss_use_topm_init:
            if int(args.faiss_topm_init) > 0:
                topm = int(args.faiss_topm_init)
            else:
                kabsmx = float(getattr(attention, 'k_abs_max', args.k_abs_max))
                topm = int(math.ceil(kabsmx * float(args.faiss_c)))
        else:
            with torch.no_grad():
                k_learned = float(attention._compute_k(int(N)).detach().cpu().item())
            topm = int(math.ceil(k_learned * float(args.faiss_c)))
        topm = max(int(args.faiss_topm_min), min(int(args.faiss_topm_max), int(topm)))

        h_idx = _get_h_idx(args.horizon)
        emb_q_shared = query_embed(torch.tensor(h_idx, device=device))  # [d]
        qa = emb_nodes[anchor_indices] + emb_q_shared.view(1, -1)
        Q_batch = attention.W_q(qa).detach()
        if getattr(attention, 'normalize_qk', False):
            Q_batch = F.normalize(Q_batch, dim=-1)
        cand_batch, _ = retriever.search(
            Q_batch,
            topm=topm,
            normalize=getattr(attention, 'normalize_qk', False),
            require_torch_gpu=bool(args.faiss_require_torch_gpu),
        )  # [B, topm]

    loss_t = 0.0
    all_context, all_query, all_target = [], [], []

    # use last-lag edges for subgraph extraction (CPU)
    ei_now_cpu = snap.edge_indices[-1]  # CPU
    for _i, anchor_b in enumerate(anchor_indices.tolist()):
        # label on CPU
        y = label_peer_quantile_future_count(
            snaps=snaps,
            t_now=snap.t,
            horizon=args.horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            k_hop=args.k_hop,
            tau_thr=tau_thr,
            subset_cache=subset_cache_cpu,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)

        emb_a = emb_nodes[anchor_b]
        h_idx = _get_h_idx(args.horizon)
        emb_q = query_embed(torch.tensor(h_idx, device=device))

        # =========================
        # Attention over full node set OR candidate-only attention (FAISS)
        # =========================
        if use_faiss:
            if cand_batch is None:
                raise RuntimeError('use_faiss=True but cand_batch is None')
            cand_ids = cand_batch[_i].view(-1)
            if args.faiss_union_khop:
                sub_nodes, _, _, _ = k_hop_subgraph(
                    node_idx=int(anchor_b),
                    num_hops=int(args.k_hop),
                    edge_index=ei_now_cpu,
                    relabel_nodes=False,
                    num_nodes=int(N),
                )
                sub_nodes = sub_nodes.to(device=device)
                cand_ids = torch.unique(torch.cat([cand_ids, sub_nodes.view(-1)], dim=0))
            if cand_ids.numel() > int(args.faiss_max_cand):
                cand_ids = cand_ids[: int(args.faiss_max_cand)]
            emb_nodes_c = emb_nodes[cand_ids]
            context, attn, soft_mask, scores, Q, K, theta, k = attention.forward_candidates(
                emb_q=emb_q,
                emb_a=emb_a,
                emb_nodes_cand= emb_nodes_c,
                node_mask=None,
                tau=tau,
                return_intermediates=True,
            )
        else:
            context, attn, soft_mask, scores, Q, K, theta, k = attention(
                emb_q=emb_q,
                emb_a=emb_a,
                emb_nodes=emb_nodes,
                node_mask=None,
                tau=tau,
                return_intermediates=True,
            )

        rep = emb_a + emb_q + context
        logits2 = head(rep)

        loss_t = loss_t + ce(logits2.view(1, 2), y_class)

        # diversity term 
        if args.beta_div > 0:
            k_hard = max(2, int(round(float(k.detach().cpu().item()))))
            loss_t = loss_t + args.beta_div * diversity_loss(attn, emb_nodes, k=k_hard, node_mask=None)

        all_context.append(context)
        all_query.append(emb_q)
        all_target.append(y_class.view(()))  # scalar

    if len(all_target) == 0:
        return None

    loss_t = loss_t / float(len(all_target))

    # mixup + metric 
    if beta_metric > 0 or beta_mixup > 0:
        ctx = torch.stack(all_context).to(device)
        qry = torch.stack(all_query).to(device)
        tgt = torch.stack(all_target).view(-1).long().to(device)

        mix_ctx, mix_tgt, tgt_i, tgt_j, mix_qry = mixup.get_mixup_samples(ctx, qry, tgt)

        mloss = metric_loss(ctx, mix_ctx, qry, mix_qry, tgt, tgt_i, tgt_j)
        if torch.is_tensor(mloss):
            mloss = mloss.mean()
        loss_t = loss_t + beta_metric * mloss

        mix_rep = mix_ctx + mix_qry
        mix_logits2 = head(mix_rep)
        mix_ce = F.cross_entropy(mix_logits2, mix_tgt, reduction="mean")
        loss_t = loss_t + beta_mixup * mix_ce

    opt.zero_grad(set_to_none=True)
    loss_t.backward()
    opt.step()

    return float(loss_t.detach().cpu().item())


# =========================
# Eval step
# =========================
@torch.inference_mode()
def eval_one_snapshot(
    *,
    args,
    snaps: List[Snap],
    snap: Snap,
    Nb: int,
    anchors_all_cpu: torch.Tensor,
    encoder: DGNNEncoder,
    head: nn.Module,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau: float,                 # attention temperature
    horizon: int,
    use_faiss: bool,
    retriever: Optional[object],
    tau_thr: float = 0.0,       # LABEL threshold
    subset_cache_cpu: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, float]]:
    encoder.eval()
    head.eval()
    attention.eval()
    query_embed.eval()

    if subset_cache_cpu is None:
        subset_cache_cpu = {}

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_from_filtered_cpu(anchors_all_cpu, k=args.anchors_eval, device=device)

    # =========================
    # FAISS (optional)
    # =========================
    cand_batch = None
    if use_faiss:
        if retriever is None:
            raise RuntimeError('use_faiss=True but retriever is None')
        valid_idx = torch.arange(N, device=device, dtype=torch.long)
        K_all = attention.W_k(emb_nodes).detach()
        retriever.build(
            K_all,
            valid_idx=valid_idx,
            normalize=getattr(attention, 'normalize_qk', False),
            require_torch_gpu=bool(args.faiss_require_torch_gpu),
        )
        with torch.no_grad():
            k_learned = float(attention._compute_k(int(N)).detach().cpu().item())
        topm = int(math.ceil(k_learned * float(args.faiss_c)))
        topm = max(int(args.faiss_topm_min), min(int(args.faiss_topm_max), int(topm)))

        h_idx = _get_h_idx(horizon)
        emb_q_shared = query_embed(torch.tensor(h_idx, device=device))
        qa = emb_nodes[anchor_indices] + emb_q_shared.view(1, -1)
        Q_batch = attention.W_q(qa).detach()
        if getattr(attention, 'normalize_qk', False):
            Q_batch = F.normalize(Q_batch, dim=-1)
        cand_batch, _ = retriever.search(
            Q_batch,
            topm=topm,
            normalize=getattr(attention, 'normalize_qk', False),
            require_torch_gpu=bool(args.faiss_require_torch_gpu),
        )

    # for attribution metrics
    snapshot_labels = []
    snapshot_anchor_idx = []
    snapshot_attn = []
    snapshot_soft_mask = []
    snapshot_k = []
    suff_list = []

    loss_sum = 0.0
    used = 0

    ei_now_cpu = snap.edge_indices[-1]  # CPU
    h_idx = _get_h_idx(horizon)

    for _i, anchor_b in enumerate(anchor_indices.tolist()):
        y = label_peer_quantile_future_count(
            snaps=snaps,
            t_now=snap.t,
            horizon=horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            k_hop=args.k_hop,
            tau_thr=tau_thr,
            subset_cache=subset_cache_cpu,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)

        emb_a = emb_nodes[anchor_b]
        emb_q = query_embed(torch.tensor(h_idx, device=device))

        if use_faiss:
            if cand_batch is None:
                raise RuntimeError('use_faiss=True but cand_batch is None')
            cand_ids = cand_batch[_i].view(-1)
            if args.faiss_union_khop:
                sub_nodes, _, _, _ = k_hop_subgraph(
                    node_idx=int(anchor_b),
                    num_hops=int(args.k_hop),
                    edge_index=ei_now_cpu,
                    relabel_nodes=False,
                    num_nodes=int(N),
                )
                sub_nodes = sub_nodes.to(device=device)
                cand_ids = torch.unique(torch.cat([cand_ids, sub_nodes.view(-1)], dim=0))
            if cand_ids.numel() > int(args.faiss_max_cand):
                cand_ids = cand_ids[: int(args.faiss_max_cand)]
            emb_nodes_c = emb_nodes[cand_ids]
            context, attn, soft_mask, scores, Q, K, theta, k = attention.forward_candidates(
                emb_q=emb_q,
                emb_a=emb_a,
                emb_nodes_cand=emb_nodes_c,
                node_mask=None,
                tau=tau,
                return_intermediates=True,
            )
        else:
            context, attn, soft_mask, scores, Q, K, theta, k = attention(
                emb_q=emb_q,
                emb_a=emb_a,
                emb_nodes=emb_nodes,
                node_mask=None,
                tau=tau,
                return_intermediates=True,
            )

        rep = emb_a + emb_q + context
        logits2 = head(rep)

        loss_sum += float(ce(logits2.view(1, 2), y_class).item())
        used += 1

        snapshot_labels.append(y_class.view(()))
        snapshot_anchor_idx.append(anchor_b)
        snapshot_attn.append(attn.detach())
        snapshot_soft_mask.append(soft_mask.detach())
        snapshot_k.append(k.detach())

        suff = compute_sufficiency(
            attention=attention,
            head=head,
            emb_q=emb_q,
            emb_a=emb_a,
            emb_nodes=emb_nodes,
            node_mask=None,
            attn=attn,
            soft_mask=soft_mask,
            k_estimate=k,
            tau=tau,
        )
        suff_list.append(suff)

    if used == 0:
        return 0.0, {
            "deletion_auc": 0.5,
            "deletion_curve": [],
            "deletion_fracs": [],
            "sufficiency": 0.0,
        }

    avg_loss = loss_sum / float(used)

    if len(snapshot_labels) == 0:
        return avg_loss, {
            "deletion_auc": 0.5,
            "deletion_curve": [],
            "deletion_fracs": [],
            "sufficiency": 0.0,
        }

    labels_batch = torch.stack(snapshot_labels)

    fracs_t, del_scores, del_auc = compute_deletion_curve(
        encoder=encoder,
        head=head,
        labels_batch=labels_batch,
        emb_q=emb_q,
        anchor_idx=snapshot_anchor_idx,
        emb_nodes_full=emb_nodes,
        X=X,
        edge_indices=eis,
        edge_weights=ews,
        attn_batch=snapshot_attn,
        soft_mask_batch=snapshot_soft_mask,
        k_batch=snapshot_k,
        removal_fracs=args.deletion_fracs,
        k_eval=args.k_eval,
    )

    metrics = {
        "deletion_auc": float(del_auc),
        "deletion_curve": del_scores.detach().cpu().tolist(),
        "deletion_fracs": fracs_t.detach().cpu().tolist(),
        "sufficiency": float(sum(suff_list) / max(1, len(suff_list))),
    }

    return avg_loss, metrics


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # COVID inputs
    p.add_argument("--nyt_csv", type=str, required=True, help="NYT us-counties.csv")
    p.add_argument("--adj_txt", type=str, required=True, help="Census county adjacency txt (pipe-delimited)")
    p.add_argument("--date_start", type=str, default="2021-04-30")
    p.add_argument("--date_end", type=str, default="2022-04-30")
    p.add_argument("--snapshot_days", type=int, default=7)
    p.add_argument("--feature_mode", type=str, default="cases_only", choices=["cases_only", "cases_deaths"])

    # split
    p.add_argument("--train_end_start", type=str, default="2022-01-30",
                   help="train last snapshot start date (inclusive); test starts next snapshot")
    p.add_argument("--holdout_snaps", type=int, default=0, help="internal holdout size (snapshots) inside train (not used in online learning, set 0)")

    p.add_argument("--model", type=str, default="TASER", choices=["DCRNN", "SEHTGNN", "TASER"])
    p.add_argument("--cpu", action="store_true")

    # temporal
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1)

    # subgraph
    p.add_argument("--k_hop", type=int, default=2)

    # threshold percentile for label (DEFAULT 0.85 per your request)
    p.add_argument("--q_thr", type=float, default=0.9)

    # model dims
    p.add_argument("--d_emb", type=int, default=64)
    p.add_argument("--dcrnn_k", type=int, default=2)

    # opt
    p.add_argument("--lr", type=float, default=1e-3)

    # sampling
    p.add_argument("--anchors_train", type=int, default=512)
    p.add_argument("--anchors_eval", type=int, default=512)
    p.add_argument("--k_eval", type=int, default=50)
    p.add_argument("--deletion_fracs", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # warmup
    p.add_argument("--warmup_snaps", type=int, default=4)
    p.add_argument("--epochs_warmup", type=int, default=1)
    p.add_argument("--anneal", type=float, default=0.9)
    p.add_argument("--print_every", type=int, default=1)

    # attention params
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--init_k_frac", type=float, default=0.05)
    p.add_argument("--k_min", type=float, default=0.01)
    p.add_argument("--k_max", type=float, default=0.2)
    p.add_argument("--k_abs_min", type=int, default=20)
    p.add_argument("--k_abs_max", type=int, default=50)

    # loss weights
    p.add_argument("--beta_div", type=float, default=0.1)
    p.add_argument("--beta_metric", type=float, default=0.5)
    p.add_argument("--beta_mixup", type=float, default=1.0)

    # -------------------------
    # FAISS (optional)
    # -------------------------
    p.add_argument('--use_faiss', action='store_true', help='Enable FAISS candidate retrieval (candidate-only attention).')
    p.add_argument('--faiss_metric', type=str, default='ip', choices=['ip', 'l2'], help='FAISS metric.')
    p.add_argument('--faiss_update_every', type=int, default=1, help='Rebuild FAISS index every R snapshots (1=every snapshot).')
    p.add_argument('--faiss_c', type=float, default=30.0, help='Candidate multiplier: topm ~= learned_k * c (after warmup).')
    p.add_argument('--faiss_topm_init', type=int, default=0, help='Warmup-only init topm (0 -> k_abs_max*c). Used only in last warmup epoch.')
    p.add_argument('--faiss_topm_min', type=int, default=64, help='Minimum topm to request from FAISS.')
    p.add_argument('--faiss_topm_max', type=int, default=4096, help='Maximum topm to request from FAISS.')
    p.add_argument('--faiss_max_cand', type=int, default=4096, help='Cap candidate set size after optional union with k-hop.')
    p.add_argument('--faiss_union_khop', action='store_true', help='Union FAISS candidates with k-hop subset.')
    p.add_argument('--faiss_require_torch_gpu', action='store_true', help='Hard-fail if faiss.contrib.torch_utils is unavailable.')

    # -------------------------
    # Sliding Window
    # -------------------------
    p.add_argument("--roll_history", type=int, default=-1,
                   help="rolling train history length in snapshots. -1=all past (expanding), H>=1=last H (sliding)")
    p.add_argument("--roll_epochs", type=int, default=1,
                   help="how many passes over the rolling history window before evaluating each snapshot")

    p.add_argument("--burnin_eval", type=int, default=0,
                   help="Skip metric computation for first B snapshots in the rolling loop (still updates model).")

    return p.parse_args()


def main():
    args = parse_args()

    # Run baseline (FAISS off) always
    warm0, run0, summ0 = train_one_run(args, use_faiss=False)

    print("\n===== Summary =====")
    print(f"No-FAISS: warmup_time={warm0:.2f}s run_time={run0:.2f}s")
    for split, m in summ0.items():
        if split != "holdout":
            print(
                f"  [{split}] avg_loss={m['avg_loss']:.6f} "
                f"deletion_auc={m['deletion_auc']:.3f} sufficiency={m['sufficiency']:.3f} "
                f"steps={m['steps']}"
            )


    # Optional: run FAISS candidate retrieval (candidate-only attention)
    if bool(getattr(args, 'use_faiss', False)):
        warm1, run1, summ1 = train_one_run(args, use_faiss=True)
        print(f"\nFAISS: warmup_time={warm1:.2f}s run_time={run1:.2f}s")
        for split, m in summ1.items():
            if split != "holdout":
                print(
                    f"  [{split}] avg_loss={m['avg_loss']:.6f} "
                    f"deletion_auc={m['deletion_auc']:.3f} sufficiency={m['sufficiency']:.3f} "
                    f"steps={m['steps']}"
                )


if __name__ == "__main__":
    main()
