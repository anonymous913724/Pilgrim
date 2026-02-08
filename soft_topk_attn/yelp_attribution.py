#!/usr/bin/env python3
"""

Yelp bipartite (business-user) DGNN + attention + mixup + metric test code
using Yelp API (yelp.py).

Task:
For anchor business b at time t, predict whether at time t+h b's review-count
is higher than quantile q (0.5 or 0.75) of the businesses that were most-recently
reviewed (within the last W months) by b's reviewers.

- Unified indexing is business first, then users (Nb+u), provided by yelp.py.
- Subgraph is k-hop on the bipartite graph (default 2-hop gives b-u-b structure).
- Loss style : binary logits (2-class CE), attention context + query, optional mixup + metric loss.

Attribution:
- Sufficiency: remove all but top-k nodes by attention and see how much prediction changes.
- Deletion: remove top-k nodes by attention and see how NDCG@k changes as we increase k (more nodes removed).

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import k_hop_subgraph

# ---- project modules ----
from attention_layer import QKOnlySoftTopKAttention
from diversity_loss_f import diversity_loss
from mixup_cap import MixupWithMemory
from metric_loss import MetricLoss
from metrics_bin_updated import ndcg_at_k_from_logits

# ---- Yelp API ----
from yelp import YelpBipartiteTemporal


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
    encoder: DGNNEncoder,
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
    Deletion curve on Yelp: remove top-k important nodes and recompute embeddings.
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

    ndcg_scores = []

    for frac in fracs_t.tolist():
        all_scores = []
        for i in range(len(anchor_idx)):
            k_estimate = k_batch[i]
            k_hard = max(1, min(int(round(float(k_estimate.detach().item()))), valid_k))
            remove_k = int(round(float(frac * k_hard)))
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
     
            context_del = torch.sum(w * emb_nodes_full, dim=0)

            rep = emb_q + context_del
            logits = head(rep)

            score = ce_margin_score(logits)
            
            all_scores.append(score)

        all_scores_t = torch.stack(all_scores)
        ndcg = ndcg_at_k_from_logits(all_scores_t, labels_batch, k=k_eval)
        ndcg_scores.append(ndcg)

    scores_t = torch.tensor(ndcg_scores, device=device)
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

        if name == "DCRNN":
            from torch_geometric_temporal.nn.recurrent import DCRNN as _DCRNN
            self.cell = _DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)
        elif name == "SEHTGNN":
            from SEHTGNN import SEHTGNN as _SEHTGNN
            self.cell = _SEHTGNN(in_channels=in_channels, out_channels=d_emb, K=K, time_window=lags)
        elif name == "TASER":
            from taser import TaserTGNNCell as _TASER
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


def pool_subgraph_mean(emb_nodes: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    return emb_nodes[subset].mean(dim=0)


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

    All computed on CPU for correctness + no device mismatch.
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
def _review_count_at(
    ds: YelpBipartiteTemporal,
    t: int,
) -> torch.Tensor:
    """
    Return count[b] = number of reviews for business b at month t.
    (counts edges u->b; if a user reviewed multiple times it counts multiple events)
    """
    src, dst, _ = ds.get_event_list(t, direction="u2b", include_edge_attr=False, device=None)
    if dst.numel() == 0:
        # will be sized by Nb at callsite once we know Nb
        return torch.empty((0,), dtype=torch.long)

    Nb = int(torch.min(src).item())  # not safe; don't use. Will pass Nb outside.
    raise RuntimeError("Do not call _review_count_at without Nb. Use _review_count_at_nb.")


@torch.no_grad()
def _review_count_at_nb(ds: YelpBipartiteTemporal, t: int, Nb: int) -> torch.Tensor:
    src, dst, _ = ds.get_event_list(t, direction="u2b", include_edge_attr=False, device=None)
    cnt = torch.zeros((Nb,), dtype=torch.long)
    if dst.numel() == 0:
        return cnt
    # dst are business ids in [0..Nb-1]
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
    Returns:
      - 1/0 label if evidence is enough
      - None if not enough evidence (caller should skip this anchor)

    Definition:
      users = reviewers of anchor_b in last recent_window months up to t_now
      peer businesses = { most recent business reviewed by each such user in that window }
      At time t_future = t_now + horizon:
        count_anchor = #reviews for anchor_b
        count_peers = #reviews for each peer business
      label = 1 if count_anchor > quantile(count_peers, q)
    """
    t_future = t_now + horizon
    if t_future >= len(ds):
        return None

    b2users, user_last_biz = _build_u2b_adjacency_for_window(ds, t_now, recent_window, Nb)

    users = b2users.get(int(anchor_b), [])
    # de-dup users
    if len(users) == 0:
        return None
    users = list(set(users))

    if len(users) < int(min_reviewers):
        return None

    peers = []
    for u in users:
        if u in user_last_biz:
            peers.append(user_last_biz[u])

    # remove anchor itself (optional; keep simple: remove if present)
    peers = [b for b in peers if b != int(anchor_b)]
    peers = list(set(peers))

    if len(peers) < int(min_peers):
        return None

    cnt_future = _review_count_at_nb(ds, t_future, Nb=Nb)

    c_anchor = int(cnt_future[int(anchor_b)].item())
    peer_counts = cnt_future[torch.tensor(peers, dtype=torch.long)]
    if peer_counts.numel() == 0:
        return None

    # quantile threshold (on CPU)
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
    Convert ds.to_dynamic_graph_temporal_signal() into our Snap list.
    Each Snap contains:
      - x_flat: [N, lags*F]
      - edge_indices/edge_weights: list length=lags (dynamic)
    """
    dgts = ds.to_dynamic_graph_temporal_signal(
        bidirectional=bidirectional,
        add_type_onehot=add_type_onehot,
        device=None,   # keep on CPU; we move per-step to GPU as needed
    )

    # infer sizes
    d0 = ds.get_hetero(0)
    Nb = int(d0["business"].x.size(0))
    Nu = int(d0["user"].x.size(0))
    N = Nb + Nu

    T = len(ds)
    snaps: List[Snap] = []
    base = lags - 1

    for t in range(base, T):
        # features stack for lags: [t-lags+1 .. t]
        feats = []
        eis = []
        ews = []
        for tt in range(t - lags + 1, t + 1):
            g = dgts[tt]  # PyG Data
            feats.append(g.x.cpu())
            eis.append(g.edge_index.cpu())
            ew = None
            # PGT uses edge_attr for snapshot Data, but DGTS stores weights separately.
            # In ds.to_dynamic_graph_temporal_signal, edge_weights become snapshot.edge_attr in Data.
            # So use g.edge_attr if present.
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                ew = g.edge_attr.view(-1).cpu()
            else:
                # fallback: all ones
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
def train_one_run(args, use_faiss: bool):
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

    metric_loss = MetricLoss(num_classes=2, d_emb=args.d_emb, code_size=args.d_emb, device=device).to(device)
    mixup = MixupWithMemory(num_classes=2, d_emb=args.d_emb, device=device)

    params = list(encoder.parameters()) + list(head.parameters()) + list(query_embed.parameters()) + list(attention.parameters()) + list(metric_loss.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    ce = nn.CrossEntropyLoss()

    # protocol
    W = max(1, min(args.warmup_snaps, S - 1))
    Hs = tuple(args.test_horizons)
    Hmax = max(Hs)

    print(f"\n=== Yelp PeerQuantile Task | model={args.model} | FAISS={'ON' if use_faiss else 'OFF'} ===")
    print(f"pt={args.pt}")
    print(f"T(raw)={len(ds)}  S(snaps)={S}  Nb={Nb} Nu={Nu} N={Nb+Nu}")
    print(f"lags={args.lags} k-hop={args.k_hop} horizon(train)={args.horizon}")
    print(f"recent_window={args.recent_window} quantile={args.quantile} (0.5=median,0.75=top-quartile)")
    print(f"evidence: min_reviewers={args.min_reviewers}, min_peers={args.min_peers}")
    print(f"warmup: W={W} epochs={args.epochs_warmup}")
    print(f"rolling horizons={Hs}")

    # ---- warmup ----
    _sync_if_cuda(device)
    t0 = _now()
    for ep in range(args.epochs_warmup):
        # annealing
        tau_ep = args.tau * (args.anneal ** ep)
        beta_metric_ep = args.beta_metric * (args.anneal ** ep)
        beta_mixup_ep = args.beta_mixup * (args.anneal ** ep)

        losses = []
        for s in range(W):
            print(f"Warmup epoch {ep+1}/{args.epochs_warmup} | snapshot {s}/{W-1} (t={snaps[s].t})")
            loss_val = train_step_one_snapshot(
                args=args,
                ds=ds,
                snap=snaps[s],
                Nb=Nb,
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
                use_faiss=use_faiss,
            )
            if loss_val is not None:
                losses.append(loss_val)
        avg = sum(losses) / max(1, len(losses))
        print(f"warmup_epoch={ep+1} avg_loss={avg:.6f} tau={tau_ep:.4f} (kept={len(losses)})")

    _sync_if_cuda(device)
    warmup_time = _now() - t0

    # ---- rolling ----
    acc = {h: {"loss": 0.0, "deletion_auc": 0.0, "sufficiency": 0.0, "n": 0} for h in Hs}
    last_del_curve_data = {}  # Store deletion curve from last snapshot

    _sync_if_cuda(device)
    t1 = _now()
    for idx in range(W, S - Hmax):
        print(f"\n--- Snapshot {idx}/{S-1} (t={snaps[idx].t}) ---")
        # online anneal
        tau_t = args.tau * (args.anneal ** (idx / max(1, S)))
        beta_metric_t = args.beta_metric * (args.anneal ** (idx / max(1, S)))
        beta_mixup_t = args.beta_mixup * (args.anneal ** (idx / max(1, S)))

        _ = train_step_one_snapshot(
            args=args,
            ds=ds,
            snap=snaps[idx],
            Nb=Nb,
            encoder=encoder,
            head=head,
            attention=attention,
            query_embed=query_embed,
            metric_loss=metric_loss,
            mixup=mixup,
            opt=opt,
            ce=ce,
            device=device,
            tau=tau_t,
            beta_metric=beta_metric_t,
            beta_mixup=beta_mixup_t,
            use_faiss=use_faiss,
        )

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
                    attention=attention,
                    query_embed=query_embed,
                    ce=ce,
                    device=device,
                    tau=tau_t,
                    horizon=h,
                    deletion_fracs=args.deletion_fracs,
                )
                acc[h]["loss"] += float(loss_h)
                acc[h]["deletion_auc"] += float(m_h["deletion_auc"])
                acc[h]["sufficiency"] += float(m_h["sufficiency"])
                acc[h]["n"] += 1
                
                # Store last snapshot's deletion curve for plotting
                if idx == S - Hmax - 1:  # Last snapshot
                    last_del_curve_data[h] = {
                        "fracs": m_h["deletion_fracs"],
                        "scores": m_h["deletion_curve"],
                        "auc": m_h["deletion_auc"],
                    }

                msg = [f"t={idx}"]
                for h in Hs:
                    n = acc[h]["n"]
                    if n > 0:
                        msg.append(
                            f"h{h}:del_AUC={acc[h]['deletion_auc'] / n:.3f},suff={acc[h]['sufficiency'] / n:.3f}"
                        )

                print("  " + " | ".join(msg))

    _sync_if_cuda(device)
    rolling_time = _now() - t1

    # Plot and save deletion curves for last snapshot
    for h, data in last_del_curve_data.items():
        if len(data["fracs"]) > 0 and len(data["scores"]) > 0:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 5))
            plt.plot(data["fracs"], data["scores"], marker="o", linewidth=2)
            plt.xlabel("Fraction of top-k nodes deleted")
            plt.ylabel(f"NDCG@{args.k_eval}")
            plt.title(f"Yelp Deletion Curve (Horizon={h})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = f"yelp_deletion_curve_h{h}_no_emba.png"
            plt.savefig(out_path, dpi=150)
            print(f"Saved deletion curve plot to {out_path}")
            plt.close()
            
            np.savez(
                f"yelp_deletion_curve_h{h}_no_emba.npz",
                fracs=np.array(data["fracs"]),
                scores=np.array(data["scores"]),
                auc=data["auc"],
            )
            print(f"Saved deletion curve data to yelp_deletion_curve_h{h}_no_emba.npz")

    # finalize
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


def _get_h_idx(h: int) -> int:
    return 0 if int(h) == 1 else 1


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
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    metric_loss: MetricLoss,
    mixup: MixupWithMemory,
    opt: torch.optim.Optimizer,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau: float,
    beta_metric: float,
    beta_mixup: float,
    use_faiss: bool,   # placeholder (kept for interface); re-add FAISS later cleanly
) -> Optional[float]:
    encoder.train()
    head.train()
    attention.train()
    query_embed.train()
    metric_loss.train()

    # build X
    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    # per-lag edges to device
    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)  # [N, d_emb]
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_business_only(Nb=Nb, k=args.anchors_train, device=device)

    loss_t = 0.0
    all_context, all_query, all_target = [], [], []

    # use last-lag edges for subgraph extraction
    ei_now = snap.edge_indices[-1]  # CPU
    for anchor_b in anchor_indices.tolist():
        # build k-hop subset on CPU, then move subset to device for indexing embeddings
        subset = _khop_subset(anchor_b, ei_now, args.k_hop, num_nodes=N).to(device=device)

        # label on CPU (safe)
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
        h_idx = _get_h_idx(args.horizon)
        emb_q = query_embed(torch.tensor(h_idx, device=device))

        # attention over full node set (simple baseline for now; swap to FAISS later)
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

        # optional diversity term (kept)
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


@torch.inference_mode()
def eval_one_snapshot(
    *,
    args,
    ds: YelpBipartiteTemporal,
    snap: Snap,
    Nb: int,
    encoder: DGNNEncoder,
    head: nn.Module,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    tau: float,
    horizon: int,
    deletion_fracs: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    encoder.eval()
    head.eval()
    attention.eval()
    query_embed.eval()

    x = snap.x_flat.to(device=device, dtype=torch.float32)
    X = x_to_batched_sequence(x, lags=args.lags)

    eis = [ei.to(device=device) for ei in snap.edge_indices]
    ews = [ew.to(device=device, dtype=torch.float32) for ew in snap.edge_weights]

    emb_nodes = encoder(X, eis, ews)
    N = emb_nodes.size(0)

    anchor_indices = _sample_anchors_business_only(Nb=Nb, k=args.anchors_eval, device=device)

    if deletion_fracs is None:
        deletion_fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    loss_sum = 0.0
    used = 0

    ei_now = snap.edge_indices[-1]  # CPU for khop
    h_idx = _get_h_idx(horizon)
    emb_q = query_embed(torch.tensor(h_idx, device=device))

    snapshot_labels = []
    snapshot_anchor_idx = []
    snapshot_attn = []
    snapshot_soft_mask = []
    snapshot_k = []
    suff_list = []

    for idx, anchor_b in enumerate(anchor_indices.tolist()):
        print(f"Evaluating anchor {idx+1}/{len(anchor_indices)}", end="\r")

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
        removal_fracs=deletion_fracs,
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

    p.add_argument("--pt", type=str, required=True, help="preprocessed Yelp .pt (from yelp_process.py)")
    p.add_argument("--model", type=str, default="SEHTGNN", choices=["DCRNN", "SEHTGNN", "TASER"])

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
    # warmup/rolling
    p.add_argument("--warmup_snaps", type=int, default=6)
    p.add_argument("--epochs_warmup", type=int, default=3)
    p.add_argument("--anneal", type=float, default=0.9)
    p.add_argument("--print_every", type=int, default=1)

    # attention params
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--init_k_frac", type=float, default=0.05)
    p.add_argument("--k_min", type=float, default=0.01)
    p.add_argument("--k_max", type=float, default=0.2)
    p.add_argument("--k_abs_min", type=int, default=10)
    p.add_argument("--k_abs_max", type=int, default=50)

    # loss weights
    p.add_argument("--beta_div", type=float, default=0.0)
    p.add_argument("--beta_metric", type=float, default=0.5)
    p.add_argument("--beta_mixup", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    # run twice : FAISS off/on placeholder
    # (currently both runs are identical; re-add FAISS block later cleanly)
    warm0, roll0, summ0 = train_one_run(args, use_faiss=False)

    print("\n===== Summary =====")
    print(f"No-FAISS: warmup_time={warm0:.2f}s rolling_time={roll0:.2f}s")
    for h, m in summ0.items():
        print(
            f"  horizon={h} avg_loss={m['avg_loss']:.6f} "
            f"del_AUC={m['deletion_auc']:.3f} suff={m['sufficiency']:.3f} "
            f"steps={m['steps']}"
        )

    # later re-enable FAISS retrieval:
    # warm1, roll1, summ1 = train_one_run(args, use_faiss=True)
    # and print both.


if __name__ == "__main__":
    main()
