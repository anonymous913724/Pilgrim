#!/usr/bin/env python3
#
"""
COVID county DGNN-only baseline attribution code

Task (COVID):
For an anchor county v at time t, define subgraph S(v) as k-hop neighborhood on the static county adjacency graph.
Define subgraph score at time t:
  score_t(v) = mean( x_t[u] for u in S(v) ) / sqrt(|S(v)|)      (density-normalized mean)
Label:
  y_t(v) = 1 if score_{t+i}(v) > tau_thr else 0
where tau_thr is computed from TRAIN only using percentile q_thr (default 0.85).

DGNN-only baseline:
  rep = DGNNEncoder(X, edge_index, edge_weight)[v]
  logits = BinaryCEHead(rep)
  loss = CE(logits, y)

Attribution:
Train the baseline DGNN model
Extract node embeddings and compute Integrated Gradients attributions for each anchor v at time t
Use the attributions to compute a top-k set of nodes for each prediction (using L2 norm of attributions to rank nodes)
Use the top-k nodes to compute sufficiency and deletion curve metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.utils import k_hop_subgraph

from soft_topk_attn.models.metrics_bin_updated import (
    f1_from_logits,
)

from soft_topk_attn.models.integrated_gradients import (
    compute_integrated_gradients,
    compute_node_importance_scores,
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
    edge_index_np = dataset.edge_index
    if isinstance(edge_index_np, np.ndarray):
        edge_index_cpu = torch.from_numpy(edge_index_np).to(dtype=torch.long)
    else:
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
                Nb=N,
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
    x_t: torch.Tensor,            # [N]
    subset_cpu: torch.Tensor,     # [|S|] on CPU
) -> float:
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
    anchors_cpu: torch.Tensor,        # [A]
    k_hop: int,
    horizon: int,
    q_thr: float,
    train_last_t: int,          # last raw t (snap.t) included in train INPUT
    lags: int,
) -> float:
    q = float(q_thr)
    q = max(0.0, min(1.0, q))

    if train_last_t - horizon < 0:
        raise ValueError("train_last_t too small for given horizon")

    edge_index_cpu = snaps[0].edge_indices[-1]
    N = int(snaps[0].Nb)

    cache: Dict[int, torch.Tensor] = {}
    scores = []

    # map raw-t -> index in snaps list: in this file snap.t equals raw t and snaps are contiguous
    # so snaps[t_f] is valid as long as t_f is in range
    for t in range(0, int(train_last_t - horizon) + 1):
        t_f = t + int(horizon)
        x_f_flat = snaps[t_f].x_flat  # CPU [N, lags*F]
        Fdim = x_f_flat.size(1) // int(lags)
        # use last-lag feature[:,0]
        x_sig = x_f_flat[:, (int(lags) - 1) * Fdim + 0].to(dtype=torch.float32)

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
    anchor_b: int,
    Nb: int,
    k_hop: int,
    tau_thr: float,           # LABEL threshold
    subset_cache: Dict[int, torch.Tensor],
    lags: int,
) -> Optional[int]:
    t_f = int(t_now) + int(horizon)
    if t_f >= len(snaps):
        return None

    edge_index_cpu = snaps[0].edge_indices[-1]
    x_f_flat = snaps[t_f].x_flat  # CPU
    Fdim = x_f_flat.size(1) // int(lags)
    x_sig = x_f_flat[:, (int(lags) - 1) * Fdim + 0].to(dtype=torch.float32)

    subset = _khop_subset_cached(anchor_b, edge_index_cpu, k_hop, num_nodes=Nb, cache=subset_cache)
    if subset.numel() <= 0:
        return None

    sc = _subgraph_score_density_normalized(x_sig, subset)
    if not math.isfinite(sc):
        return None

    return 1 if float(sc) > float(tau_thr) else 0


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
    plt.ylabel("F1")
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
    subset_size = int(subset.numel())
    if subset_size == 0:
        return 0.0

    k_frac = max(0.0, min(1.0, float(k_frac)))
    k_hard = max(1, int(round(k_frac * subset_size)))
    
    # Rank nodes by IG importance
    node_scores = compute_node_importance_scores(attributions, aggregation='l2')
    subset_scores = node_scores[subset]
    sorted_indices = torch.argsort(subset_scores, descending=True)
    topk_indices = sorted_indices[:k_hard]
    topk_nodes = subset[topk_indices]

    ctx_full = _context_from_subset(emb_nodes, subset)
    ctx_topk = _context_from_subset(emb_nodes, topk_nodes)

    score_full = torch.sigmoid(ce_margin_score(head(emb_a + ctx_full)))
    score_topk = torch.sigmoid(ce_margin_score(head(emb_a + ctx_topk)))
    return float(abs(score_full - score_topk))


@torch.inference_mode()
def compute_deletion_curve(
    *,
    head: nn.Module,
    emb_nodes: torch.Tensor,
    anchor_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    removal_fracs: List[float],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Deletion curve with F1 using Integrated Gradients attributions.
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

    f1_scores = []
    for frac in fracs_t.tolist():
        scores = []
        labels = []
        for emb_a, subset, attributions, y_class in anchor_info:
            # Rank nodes by IG importance (L2 norm of attributions)
            node_scores = compute_node_importance_scores(attributions, aggregation='l2')
            # Get subset node scores
            subset_scores = node_scores[subset]
            sorted_indices = torch.argsort(subset_scores, descending=True)
            
            subset_size = 26
            remove_k = int(round(float(frac * subset_size)))
            remove_k = min(max(remove_k, 0), subset_size)

            keep_indices = sorted_indices[remove_k:]
            keep_nodes = subset[keep_indices]

            ctx = _context_from_subset(emb_nodes, keep_nodes)
            rep = emb_a + ctx
            logits2 = head(rep)
            scores.append(ce_margin_score(logits2))
            labels.append(y_class.float())

        scores_t = torch.stack(scores)
        labels_t = torch.stack(labels).squeeze()  # Ensure 1D tensor
        f1 = f1_from_logits(scores_t, labels_t)
        f1_scores.append(torch.tensor(f1, device=device))

    scores_t = torch.stack(f1_scores)
    auc = torch.trapz(scores_t, fracs_t).item()
    return fracs_t, scores_t, auc


# =========================
# Split helpers
# =========================
def _find_last_t_with_start_leq(meta, date_str: str) -> int:
    import pandas as pd
    d = pd.to_datetime(date_str)
    last = -1
    for i, s in enumerate(meta.snapshot_starts):
        if s <= d:
            last = i
    return int(last)


def _sample_anchors_from_filtered_cpu(anchors_all_cpu: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    a = anchors_all_cpu.to(device=device)
    if k is None or k < 0 or k >= a.numel():
        return a
    perm = torch.randperm(a.numel(), device=device)
    return a[perm[:k]]


# =========================
# Train / eval (DGNN-only)
# =========================
def train_one_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset, meta = build_nyt_covid_static_graph_temporal_signal(
        nyt_us_counties_csv=args.nyt_csv,
        census_adjacency_txt=args.adj_txt,
        date_start=args.date_start,
        date_end=args.date_end,
        snapshot_days=args.snapshot_days,
        feature_mode=args.feature_mode,
    )

    snaps, N = build_snaps_from_covid_api(dataset, meta, lags=args.lags)
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

    # -------------------------
    # Anchor filtering (degree==0 removed) - CPU once
    # -------------------------
    edge_index_cpu = snaps[0].edge_indices[-1]
    deg = torch.zeros((N,), dtype=torch.long)
    deg.index_add_(0, edge_index_cpu[0], torch.ones((edge_index_cpu.size(1),), dtype=torch.long))
    keep = (deg > 0)
    anchors_all_cpu = torch.nonzero(keep, as_tuple=False).view(-1).to(dtype=torch.long)

    # -------------------------
    # split (paper train/test), but rolling uses all chronologically
    # -------------------------
    train_last_raw = _find_last_t_with_start_leq(meta, args.train_end_start)
    test_first_raw = train_last_raw + 1

    base_raw = snaps[0].t
    last_raw = snaps[-1].t

    train_last_raw = min(int(train_last_raw), int(last_raw))
    test_first_raw = min(int(test_first_raw), int(last_raw) + 1)

    train_ids = [i for i, sp in enumerate(snaps) if int(sp.t) <= int(train_last_raw)]
    test_ids = [i for i, sp in enumerate(snaps) if int(sp.t) >= int(test_first_raw)]

    # need labels at t+horizon, so exclude tail without future
    def _filter_valid(ids: List[int]) -> List[int]:
        out = []
        for i in ids:
            t_raw = int(snaps[i].t)
            if (t_raw + int(args.horizon)) <= int(last_raw):
                out.append(i)
        return out

    train_ids = _filter_valid(train_ids)
    test_ids = _filter_valid(test_ids)

    if len(train_ids) == 0:
        raise RuntimeError("train_ids is empty after filtering.")

    train_last_raw_effective = max(int(snaps[i].t) for i in train_ids)

    tau_thr = _compute_tau_thr_from_train(
        snaps=snaps,
        anchors_cpu=anchors_all_cpu,
        k_hop=args.k_hop,
        horizon=args.horizon,
        q_thr=args.q_thr,
        train_last_t=train_last_raw_effective,
        lags=args.lags,
    )

    print(f"\n=== COVID Hotspot Exceedance Task | DGNN-only baseline | model={args.model} ===")
    print(f"nyt_csv={args.nyt_csv}")
    print(f"adj_txt={args.adj_txt}")
    print(f"T(raw)={len(dataset.features)}  S(snaps)={S}  N(counties)={N}")
    print(f"lags={args.lags} k-hop={args.k_hop} horizon={args.horizon}")
    print(f"train_end_start={args.train_end_start}  (paper test starts next snapshot)")
    print(f"anchors kept (deg>0)={int(anchors_all_cpu.numel())}/{N}")
    print(f"tau_thr(label) percentile={args.q_thr:.2f} computed on TRAIN = {tau_thr:.6f}")

    def _rolling_history_ids(all_ids_sorted: List[int], cur_idx: int, history: int) -> List[int]:
        prev_ids = [x for x in all_ids_sorted if x < cur_idx]
        if int(history) < 0:
            return prev_ids
        if int(history) == 0:
            return []
        return prev_ids[-int(history):]

    # ---- rolling / prequential mode ----
    print("\n=== Rolling / Prequential (train -> test) ===")
    _sync_if_cuda(device)
    t1 = _now()

    all_ids_sorted = sorted(list(set(train_ids + test_ids)))
    train_set = set(train_ids)
    test_set = set(test_ids)

    # Find last snapshot with valid future labels
    last_valid_idx = -1
    for idx in all_ids_sorted:
        t_f = int(snaps[idx].t) + int(args.horizon)
        if t_f <= int(snaps[-1].t):
            last_valid_idx = idx

    subset_cache_cpu: Dict[int, torch.Tensor] = {}

    for ii, idx in enumerate(all_ids_sorted):
        hist_ids = _rolling_history_ids(all_ids_sorted, cur_idx=idx, history=int(args.roll_history))

        # 1) train on history window (if any)
        if len(hist_ids) > 0:
            for _ep in range(int(args.roll_epochs)):
                for jdx in hist_ids:
                    train_loss = train_step_one_snapshot(
                        args=args,
                        snaps=snaps,
                        snap=snaps[jdx],
                        Nb=N,
                        anchors_all_cpu=anchors_all_cpu,
                        encoder=encoder,
                        head=head,
                        opt=opt,
                        ce=ce,
                        device=device,
                        tau_thr=tau_thr,
                        subset_cache_cpu=subset_cache_cpu,
                    )

                    print(f"t={idx} | train_loss={train_loss:.6f}" if train_loss is not None else f"t={idx} | train_loss=None")
        
        # 2) eval only on the last valid snapshot
        if idx != last_valid_idx:
            continue

        print(f"beginning evaluation on test snapshot t={snaps[idx].t} (last with valid future labels) ...")

        loss_h, m_h = eval_one_snapshot(
            args=args,
            snaps=snaps,
            snap=snaps[idx],
            Nb=N,
            anchors_all_cpu=anchors_all_cpu,
            encoder=encoder,
            head=head,
            ce=ce,
            device=device,
            tau_thr=tau_thr,
            subset_cache_cpu=subset_cache_cpu,
        )

        print(
            f"\nFinal evaluation at t={snaps[idx].t}:\n"
            f"  loss={float(loss_h):.6f}  "
            f"DEL-AUC={float(m_h['deletion_auc']):.4f}  "
            f"SUFF={float(m_h['sufficiency']):.4f}"
        )

    def _finalize(m_h: Dict[str, float]) -> Dict[str, float]:
        return {
            "avg_loss": m_h.get("loss", float(loss_h)),
            "deletion_auc": m_h["deletion_auc"],
            "sufficiency": m_h["sufficiency"],
            "steps": 1,
        }

    res_train = {"avg_loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "steps": 0}
    if last_valid_idx >= 0:
        res_test = _finalize(m_h)
    else:
        res_test = {"avg_loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "steps": 0}

    _sync_if_cuda(device)
    run_time = _now() - t1
    return run_time, {"train": res_train, "test": res_test}


# =========================
# Train step (DGNN-only)
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
    opt: torch.optim.Optimizer,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau_thr: float = 0.0,         # LABEL threshold
    subset_cache_cpu: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[float]:
    encoder.train()
    head.train()

    if subset_cache_cpu is None:
        subset_cache_cpu = {}

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)  # [N, d_emb]
    anchor_indices = _sample_anchors_from_filtered_cpu(anchors_all_cpu, k=args.anchors_train, device=device)

    loss_t = 0.0
    used = 0

    for anchor_b in anchor_indices.tolist():
        y = label_peer_quantile_future_count(
            snaps=snaps,
            t_now=snap.t,
            horizon=args.horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            k_hop=args.k_hop,
            tau_thr=tau_thr,
            subset_cache=subset_cache_cpu,
            lags=args.lags,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)
        logits2 = head(emb_nodes[anchor_b])  # [2]
        loss_t = loss_t + ce(logits2.view(1, 2), y_class)
        used += 1

    if used == 0:
        return None

    loss_t = loss_t / float(used)

    opt.zero_grad(set_to_none=True)
    loss_t.backward()
    opt.step()

    return float(loss_t.detach().cpu().item())


# =========================
# Eval step (DGNN-only)
# =========================
def eval_one_snapshot(
    *,
    args,
    snaps: List[Snap],
    snap: Snap,
    Nb: int,
    anchors_all_cpu: torch.Tensor,
    encoder: DGNNEncoder,
    head: nn.Module,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau_thr: float = 0.0,       # LABEL threshold
    subset_cache_cpu: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, float]]:
    encoder.eval()
    head.eval()

    if subset_cache_cpu is None:
        subset_cache_cpu = {}

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_from_filtered_cpu(anchors_all_cpu, k=args.anchors_eval, device=device)

    anchor_info = []

    loss_sum = 0.0
    used = 0

    for anchor_b in anchor_indices.tolist():
        y = label_peer_quantile_future_count(
            snaps=snaps,
            t_now=snap.t,
            horizon=args.horizon,
            anchor_b=anchor_b,
            Nb=Nb,
            k_hop=args.k_hop,
            tau_thr=tau_thr,
            subset_cache=subset_cache_cpu,
            lags=args.lags,
        )
        if y is None:
            continue

        y_class = torch.tensor([int(y)], device=device, dtype=torch.long)

        # Get subset for this anchor
        subset = _khop_subset_cached(
            anchor=anchor_b,
            edge_index_cpu=snap.edge_indices[-1],
            k_hop=args.k_hop,
            num_nodes=N,
            cache=subset_cache_cpu,
        ).to(device=device)

        emb_a = emb_nodes[anchor_b]
        ctx = _context_from_subset(emb_nodes, subset)

        rep = emb_a + ctx
        logits2 = head(rep)
        loss_sum += float(ce(logits2.view(1, 2), y_class).item())
        used += 1

        # Integrated Gradients attribution
        def classification_head(agg_embedding):
            logits = head(emb_a + agg_embedding)
            return ce_margin_score(logits)
        
        # Create attention weights that only consider the subset
        attention_weights = torch.zeros(N, device=device)
        if subset.numel() > 0:
            attention_weights[subset] = 1.0 / subset.numel()  # uniform attention over subset nodes
        
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
        fracs_t, del_scores, del_auc = compute_deletion_curve(
            head=head,
            emb_nodes=emb_nodes,
            anchor_info=anchor_info,
            removal_fracs=args.deletion_fracs,
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
        out_path = f"covid_baseline_deletion_curve_ig_h{args.horizon}.png"
        plot_deletion_curve(
            fracs=metrics["deletion_fracs"],
            scores=metrics["deletion_curve"],
            title=f"COVID Baseline Deletion Curve (IG) h={args.horizon}",
            out_path=out_path,
        )
        
        npz_path = f"covid_baseline_deletion_curve_ig_h{args.horizon}.npz"
        np.savez(
            npz_path,
            fracs=np.array(metrics["deletion_fracs"]),
            scores=np.array(metrics["deletion_curve"]),
            auc=metrics["deletion_auc"],
        )

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

    # split (paper)
    p.add_argument("--train_end_start", type=str, default="2022-01-30",
                   help="train last snapshot start date (inclusive); test starts next snapshot")

    p.add_argument("--model", type=str, default="TASER", choices=["DCRNN", "SEHTGNN", "TASER"])
    p.add_argument("--cpu", action="store_true")

    # temporal
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1)

    # subgraph
    p.add_argument("--k_hop", type=int, default=2)

    # threshold percentile for label
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
    p.add_argument("--suff_k_frac", type=float, default=1.0)
    p.add_argument("--save_del_curves", action="store_true", default=True)

    # rolling
    p.add_argument("--roll_history", type=int, default=-1,
                   help="rolling train history length in snapshots. -1=all past (expanding), H>=1=last H (sliding), 0=no training")
    p.add_argument("--roll_epochs", type=int, default=1,
                   help="how many passes over the rolling history window before evaluating each snapshot")

    p.add_argument("--print_every", type=int, default=1)

    p.add_argument("--burnin_eval", type=int, default=0,
                   help="Skip metric computation for first B snapshots in the rolling loop (still updates model).")

    return p.parse_args()


def main():
    args = parse_args()
    run_time, summ = train_one_run(args)

    print("\n===== Summary =====")
    print(f"run_time={run_time:.2f}s")
    for split, m in summ.items():
        print(
            f"  [{split}] avg_loss={m['avg_loss']:.6f} "
            f"DEL-AUC={m['deletion_auc']:.3f} "
            f"SUFF={m['sufficiency']:.3f} "
            f"steps={m['steps']}"
        )




if __name__ == "__main__":
    main()
