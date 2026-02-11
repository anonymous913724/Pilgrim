#!/usr/bin/env python3
"""
COVID county DGNN-only baseline test code

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
    binary_auc_from_logits,
    roc_auc_from_logits,
    ap_from_logits,
    precision_at_k_from_logits,
    recall_at_k_from_logits,
    f1_from_logits,
    ndcg_at_k_from_logits,
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
    anchors_cpu: torch.Tensor,
    k_hop: int,
    horizon: int,
    q_thr: float,
    train_last_t: int,
    start_t: int = 0,
) -> float:
    q = float(q_thr)
    q = max(0.0, min(1.0, q))

    start_t = int(max(0, start_t))
    train_last_t = int(train_last_t)

    if train_last_t - horizon < start_t:
        raise ValueError("tau window too small for given horizon")

    edge_index_cpu = snaps[0].edge_indices[-1]
    N = int(snaps[0].Nb)

    cache: Dict[int, torch.Tensor] = {}
    scores = []

    for t in range(start_t, int(train_last_t - horizon) + 1):
        t_f = t + int(horizon)
        x_f_flat = snaps[t_f].x_flat
        x_sig = x_f_flat[:, 0].to(dtype=torch.float32)

        for a in anchors_cpu.tolist():
            subset = _khop_subset_cached(a, edge_index_cpu, k_hop, num_nodes=N, cache=cache)
            sc = _subgraph_score_density_normalized(x_sig, subset)
            if math.isfinite(sc):
                scores.append(sc)

    if len(scores) == 0:
        raise RuntimeError("No scores collected for tau_thr computation.")

    scores_t = torch.tensor(scores, dtype=torch.float32)
    return float(torch.quantile(scores_t, q).item())


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

def _sample_anchors_from_filtered_cpu(
    anchors_all_cpu: torch.Tensor,
    k: int,
    device: torch.device,
    *,
    t_key: int,
    mode: str,
    base_seed: int,
    cache_cpu: Optional[Dict[Tuple[str, int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    # keep Yelp style: sample by randperm
    cache_key = (str(mode), int(t_key), int(k))
    if cache_cpu is not None and cache_key in cache_cpu:
        return cache_cpu[cache_key].to(device=device)

    a = anchors_all_cpu.to(device=device)
    if k is None or k < 0 or k >= a.numel():
        out = a
    else:
        # per-snapshot deterministic generator (independent of call order)
        g = torch.Generator(device=device)
        g.manual_seed(int(base_seed) + 1000003 * int(t_key) + (1 if mode == "train" else 2))
        perm = torch.randperm(a.numel(), device=device, generator=g)
        out = a[perm[:k]]

    if cache_cpu is not None:
        cache_cpu[cache_key] = out.detach().cpu()
    return out

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

    print(f"\n=== COVID Hotspot Exceedance Task | DGNN-only baseline | model={args.model} ===")
    print(f"nyt_csv={args.nyt_csv}")
    print(f"adj_txt={args.adj_txt}")
    print(f"T(raw)={len(dataset.features)}  S(snaps)={S}  N(counties)={N}")
    print(f"lags={args.lags} k-hop={args.k_hop} horizon={args.horizon}")
    print(f"train_end_start={args.train_end_start}  (paper test starts next snapshot)")
    print(f"anchors kept (deg>0)={int(anchors_all_cpu.numel())}/{N}")

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

    acc_train = {"loss": 0.0, "auc": 0.0, "roc_auc": 0.0, "ap": 0.0, "precision@k": 0.0, "recall@k": 0.0, "F1": 0.0, "ndcg@k": 0.0, "n": 0}
    acc_test  = {"loss": 0.0, "auc": 0.0, "roc_auc": 0.0, "ap": 0.0, "precision@k": 0.0, "recall@k": 0.0, "F1": 0.0, "ndcg@k": 0.0, "n": 0}

    subset_cache_cpu: Dict[int, torch.Tensor] = {}
    anchor_cache_cpu: Dict[Tuple[str, int, int], torch.Tensor] = {}

    for ii, idx in enumerate(all_ids_sorted):
        cur_raw_t = int(snaps[idx].t)
        # rolling tau_thr (needs at least horizon snapshots)
        if cur_raw_t < int(args.horizon):
            tau_thr = float(args.tau_thr_init)  # or reuse previous tau_thr
        else:
            start_t = max(0, cur_raw_t - int(args.tau_roll_window) + 1)
            start_t = min(start_t, cur_raw_t - int(args.horizon))  # <-- KEY clamp
            tau_thr = _compute_tau_thr_from_train(
                snaps=snaps,
                anchors_cpu=anchors_all_cpu,
                k_hop=args.k_hop,
                horizon=args.horizon,
                q_thr=args.q_thr,
                train_last_t=cur_raw_t,
                start_t=start_t,
            )

        hist_ids = _rolling_history_ids(all_ids_sorted, cur_idx=idx, history=int(args.roll_history))

        # 2) eval current idx
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
            anchor_cache_cpu=anchor_cache_cpu,
        )

        if idx in train_set:
            acc = acc_train
            name = "train"
        else:
            acc = acc_test
            name = "test"

        acc["loss"] += float(loss_h)
        acc["auc"] += float(m_h["auc"])
        acc["roc_auc"] += float(m_h["roc_auc"])
        acc["ap"] += float(m_h["ap"])
        acc["precision@k"] += float(m_h["precision@k"])
        acc["recall@k"] += float(m_h["recall@k"])
        acc["F1"] += float(m_h["F1"])
        acc["ndcg@k"] += float(m_h["ndcg@k"])
        acc["n"] += 1

        if (ii % max(1, args.print_every)) == 0 and name == "train":
            print(
                f"  t={snaps[idx].t:<3d}  loss={float(loss_h):.4f}  "
                f"AUC={float(m_h['auc']):.4f}  "
                f"ROC-AUC={float(m_h['roc_auc']):.4f}  AP={float(m_h['ap']):.4f}  "
                f"F1={float(m_h['F1']):.4f}  "
                f"P@k={float(m_h['precision@k']):.4f}  R@k={float(m_h['recall@k']):.4f}  "
                f"NDCG@k={float(m_h['ndcg@k']):.4f}"
            )

        # 1) train on history window (if any)
        if len(hist_ids) > 0:
            for _ep in range(int(args.roll_epochs)):
                for jdx in hist_ids:
                    _ = train_step_one_snapshot(
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
                        anchor_cache_cpu=anchor_cache_cpu,
                    )
        # 2) (optional) burn-in: skip metrics for first B steps
        if ii < int(args.burnin_eval):
            continue



    def _finalize(acc: Dict[str, float]) -> Dict[str, float]:
        n = int(acc["n"])
        if n <= 0:
            return {"avg_loss": 0.0, "auc": 0.5, "roc_auc": 0.5, "ap": 0.0, "precision@k": 0.0, "recall@k": 0.0, "F1": 0.0, "ndcg@k": 0.0, "steps": 0}
        return {
            "avg_loss": acc["loss"] / n,
            "auc": acc["auc"] / n,
            "roc_auc": acc["roc_auc"] / n,
            "ap": acc["ap"] / n,
            "precision@k": acc["precision@k"] / n,
            "recall@k": acc["recall@k"] / n,
            "F1": acc["F1"] / n,
            "ndcg@k": acc["ndcg@k"] / n,
            "steps": n,
        }

    res_train = _finalize(acc_train)
    res_test = _finalize(acc_test)

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
    anchor_cache_cpu: Optional[Dict[Tuple[str, int, int], torch.Tensor]] = None,
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
    anchor_indices = _sample_anchors_from_filtered_cpu(
        anchors_all_cpu,
        k=int(args.anchors_train),
        device=device,
        t_key=int(snap.t),
        mode="train",
        base_seed=int(args.seed),
        cache_cpu=anchor_cache_cpu,
    )

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
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau_thr: float = 0.0,       # LABEL threshold
    subset_cache_cpu: Optional[Dict[int, torch.Tensor]] = None,
    anchor_cache_cpu: Optional[Dict[Tuple[str, int, int], torch.Tensor]] = None,
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

    anchor_indices = _sample_anchors_from_filtered_cpu(
        anchors_all_cpu,
        k=int(args.anchors_eval),
        device=device,
        t_key=int(snap.t),
        mode="eval",
        base_seed=int(args.seed),
        cache_cpu=anchor_cache_cpu,
    )

    scores_cpu = []
    labels_cpu = []
    snapshot_scores = torch.full((Nb,), float("-inf"), device=device)
    snapshot_labels = torch.zeros((Nb,), device=device)

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

        logits2 = head(emb_nodes[anchor_b])
        loss_sum += float(ce(logits2.view(1, 2), y_class).item())
        used += 1

        score = float(ce_margin_score(logits2).detach().cpu().item())
        scores_cpu.append(score)
        labels_cpu.append(float(int(y)))

        snapshot_scores[anchor_b] = torch.tensor(score, device=device)
        snapshot_labels[anchor_b] = float(int(y))

    if used == 0:
        return 0.0, {"auc": 0.5, "roc_auc": 0.5, "ap": 0.0, "precision@k": 0.0, "recall@k": 0.0, "F1": 0.0, "ndcg@k": 0.0}

    avg_loss = loss_sum / float(used)
    scores_cpu_t = torch.tensor(scores_cpu, dtype=torch.float32)
    labels_cpu_t = torch.tensor(labels_cpu, dtype=torch.float32)

    metrics = {
        "auc": binary_auc_from_logits(scores_cpu_t, labels_cpu_t),
        "roc_auc": roc_auc_from_logits(scores_cpu_t, labels_cpu_t),
        "ap": ap_from_logits(scores_cpu_t, labels_cpu_t),
        "precision@k": precision_at_k_from_logits(scores_cpu_t, labels_cpu_t, k=args.k_eval),
        "recall@k": recall_at_k_from_logits(scores_cpu_t, labels_cpu_t, k=args.k_eval),
        "F1": f1_from_logits(scores_cpu_t, labels_cpu_t),
        "ndcg@k": ndcg_at_k_from_logits(scores_cpu_t, labels_cpu_t, k=args.k_eval),
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
    p.add_argument("--train_end_start", type=str, default="2022-04-30",
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
    p.add_argument("--tau_roll_window", type=int, default=8,
                        help="rolling window size (in snapshots) to compute tau_thr")
    p.add_argument("--tau_thr_init", type=float, default=0.0,
                        help="fallback tau_thr used before rolling window is valid")

    # model dims
    p.add_argument("--d_emb", type=int, default=64)
    p.add_argument("--dcrnn_k", type=int, default=2)

    # opt
    p.add_argument("--lr", type=float, default=1e-3)

    # sampling
    p.add_argument("--anchors_train", type=int, default=512)
    p.add_argument("--anchors_eval", type=int, default=512)
    p.add_argument("--k_eval", type=int, default=50)

    p.add_argument("--seed", type=int, default=0)

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

    import random
    import numpy as np

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    run_time, summ = train_one_run(args)

    print("\n===== Summary =====")
    print(f"run_time={run_time:.2f}s")
    for split, m in summ.items():
        if split == "train":
            print(
                f"  [{split}] avg_loss={m['avg_loss']:.6f} "
                f"AUC={m['auc']:.3f} "
                f"ROC-AUC={m['roc_auc']:.3f} Average Precision={m['ap']:.3f} "
                f"F1={m['F1']:.3f} "
                f"Precision@{args.k_eval}={m['precision@k']:.3f} Recall@{args.k_eval}={m['recall@k']:.3f} "
                f"NDCG@{args.k_eval}={m['ndcg@k']:.3f} "
                f"steps={m['steps']}"
            )




if __name__ == "__main__":
    main()
