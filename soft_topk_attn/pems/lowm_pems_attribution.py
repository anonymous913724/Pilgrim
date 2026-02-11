
#   Node-level prediction task with attribution analysis.

from typing import Optional, Tuple, List


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from soft_topk_attn.models.metrics_bin_updated import f1_from_logits

from soft_topk_attn.data.pems_bay import load_pemsbay_from_npy
from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.diversity_loss_f import diversity_loss
from soft_topk_attn.models.mixup import MixupWithMemory
from soft_topk_attn.models.metric_loss import MetricLoss

# =========================
# MODEL CHOICE
# =========================
# Choose one of: "DCRNN", "SEHTGNN", "TASER"
MODEL_NAME = "DCRNN"

# TASER adapter settings
TASER_NUM_NEIGHBORS = 16
TASER_DIM_TIME = 16
TASER_ATT_HEAD = 4

# =========================
# SETTINGS
# =========================
HORIZON = 1              # set to 1 or 3 (predict 1-step or 3-step future)
LAGS = 12
D_EMB = 32
DCRNN_K = 2

THRESHOLD = 60.0
TARGET_CHANNEL = 0
ADJ_THRESHOLD = 0.0

EPOCHS = 1
LR = 1e-3

K_EVAL = 50

# attention, mixup, metric loss settings
INIT_K_FRAC = 0.05
K_MIN = 0.01
K_MAX = 0.1
TAU = 0.05
BETA_DIV = 0.1
BETA_METRIC = 0.01
BETA_MIXUP = 1.0

# batching (to reduce memory)
ANCHORS_PER_SNAPSHOT_TRAIN = 32
ANCHORS_PER_SNAPSHOT_EVAL = 64
MAX_EVAL_ANCHORS = 256

# attribution metrics settings
DELETION_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# split
TRAIN_RATIO = 0.95  # 20% train, 80% test
TOTAL_RATIO = 0.025  # total % data for train+test
ADJ_PATH = "data/pems_adj_mat.npy"
VALUES_PATH = "data/pems_node_values.npy"

def _to_tensor(x, device, dtype=torch.float):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def x_to_batched_sequence(x: torch.Tensor, lags: int) -> torch.Tensor:
    # supports flattened [N, lags*F]
    if x.dim() == 2:
        N, D = x.shape
        if D % lags == 0 and D != lags:
            Fdim = D // lags
            return x.view(N, lags, Fdim).permute(1, 0, 2).unsqueeze(0)
    if x.dim() == 3:
        N, a, b = x.shape
        if b == lags:
            return x.permute(2, 0, 1).unsqueeze(0)
        if a == lags:
            return x.permute(1, 0, 2).unsqueeze(0)
    raise ValueError(f"Unrecognized x shape {tuple(x.shape)} for lags={lags}")


def ce_margin_score(logits2: torch.Tensor) -> torch.Tensor:
    return logits2[..., 1] - logits2[..., 0]


class DCRNNEncoder(nn.Module):
    # Encoder that can use different DGNN backbones.
    # MODEL_NAME choices: DCRNN, SEHTGNN, TASER
    def __init__(self, in_channels: int, d_emb: int, K: int = 2):
        super().__init__()
        name = str(MODEL_NAME).strip().upper()

        if name == "DCRNN":
            from torch_geometric_temporal.nn.recurrent import DCRNN as _DCRNN
            self.cell = _DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)

        elif name == "SEHTGNN":
            from soft_topk_attn.models.SEHTGNN import SEHTGNN as _SEHTGNN
            self.cell = _SEHTGNN(in_channels=in_channels, out_channels=d_emb, K=K, time_window=LAGS)

        elif name == "TASER":
            from soft_topk_attn.models.taser import TaserTGNNCell as _TASER
            self.cell = _TASER(
                in_channels=in_channels,
                out_channels=d_emb,
                num_neighbors=TASER_NUM_NEIGHBORS,
                dim_time=TASER_DIM_TIME,
                att_head=TASER_ATT_HEAD,
                dropout=0.0,
                time_enc="learnable",
            )
        else:
            raise ValueError(f"Unknown MODEL_NAME={MODEL_NAME!r}. Use 'DCRNN', 'SEHTGNN', or 'TASER'.")

    def forward(self, X, edge_index, edge_weight=None):
        # X: [1, L, N, F]
        
        # For SEHTGNN (avoid autograd error): reset per window if available
        if hasattr(self.cell, "reset_state"):
            self.cell.reset_state()

        H = None
        for t in range(X.size(1)):
            x_t = X[0, t]  # [N, F]
            try:
                H = self.cell(x_t, edge_index, edge_weight, H)
            except TypeError:
                H = self.cell(x_t, edge_index, edge_weight)
        return H

class BinaryCEHead(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.lin(rep)

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
    Compute sufficiency metric: |f(G) - f(G_Sk)|
    where G is the full graph and G_Sk is the induced subgraph with only top-k nodes.
    
    Smaller is better - prediction should remain stable using only evidence nodes.
    
    Returns:
        Absolute difference between full graph prediction and induced subgraph prediction.
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
    prob_full = torch.sigmoid(score_full).item()
    
    # Rank nodes by importance (soft mask × attention)
    importance = (soft_mask * attn) * base_mask.to(dtype=soft_mask.dtype)
    order = torch.argsort(importance, descending=True)
    
    valid_k = int(base_mask.sum().item())
    k_hard = max(1, min(int(round(float(k_estimate.detach().item()))), valid_k))
    
    # Create mask for induced subgraph (only top-k nodes)
    induced_mask = torch.zeros_like(base_mask)
    induced_mask[order[:k_hard]] = True
    induced_mask = induced_mask & base_mask  # Respect original mask
    
    # Induced subgraph prediction
    context_induced, _, _, _, _, _, _, _ = attention(
        emb_q=emb_q,
        emb_a=emb_a,
        emb_nodes=emb_nodes,
        node_mask=induced_mask,
        tau=tau,
        return_intermediates=True,
    )
    logits_induced = head(emb_q + context_induced)
    prob_induced = torch.sigmoid(ce_margin_score(logits_induced)).item()

    # Sufficiency: smaller is better
    sufficiency = abs(prob_full - prob_induced)
    
    return sufficiency


@torch.inference_mode()
def compute_deletion_curve(
    encoder: DCRNNEncoder,
    head: nn.Module,
    labels_batch: torch.Tensor,
    emb_q: torch.Tensor,
    anchor_idx: torch.Tensor,
    emb_nodes_full: Optional[torch.Tensor],
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    node_mask: Optional[torch.Tensor],
    attn_batch: List[torch.Tensor],
    soft_mask_batch: List[torch.Tensor],
    k_batch: List[torch.Tensor],
    removal_fracs: List[float],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Recomputes F1 score after progressively deleting the most important nodes.
    Computes a deletion score for each anchor index for each k fraction.

    x-axis: fraction of the (hard) top-k neighbors removed
    y-axis: F1 score after deletion (higher is better; should drop as nodes are removed)
    
    returns: (fractions tensor [m], f1_scores tensor [m], area-under-curve float)
    """

    device = X.device
    base_mask = torch.ones(X.size(2), dtype=torch.bool, device=device) if node_mask is None else node_mask.bool()
    valid_k = int(base_mask.sum().item())
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    
    # Prepare fractions
    fracs_t = torch.as_tensor(removal_fracs, device=device, dtype=torch.float)
    if fracs_t.numel() == 0:
        fracs_t = torch.tensor([0.0], device=device)
    if float(fracs_t.min()) > 0.0:
        fracs_t = torch.cat([torch.zeros(1, device=device), fracs_t])
    fracs_t = torch.clamp(fracs_t, 0.0, 1.0)
    fracs_t, _ = torch.sort(fracs_t)

    # compute the ranking order once per anchor and reuse for all fractions
    order = []
    for i in range(len(soft_mask_batch)):
        attn = attn_batch[i]
        soft_mask = soft_mask_batch[i]
        importance = (soft_mask * attn) * base_mask.to(dtype=soft_mask.dtype)
        order_i = torch.argsort(importance, descending=True)
        order.append(order_i)

    # For each fraction, compute F1 across all anchors
    f1_scores = []

    for frac in fracs_t.tolist():
        all_scores = []
        
        # Compute logits for each anchor with this deletion fraction
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

                # Build a pruned graph and recompute embeddings before attention
                pruned_edge_mask = mask[edge_src] & mask[edge_dst]
                pruned_edge_index = edge_index[:, pruned_edge_mask]
                pruned_edge_weight = edge_weight[pruned_edge_mask] if edge_weight is not None else None

                mask_f = mask.to(dtype=X.dtype).view(1, 1, -1, 1)
                X_masked = X * mask_f

                emb_nodes_del = encoder(X_masked, pruned_edge_index, pruned_edge_weight)
            # use the original anchor embedding in case it was masked out
            emb_a = emb_nodes_del[anchor_idx[i]]

            # Manually compute context vector using precomputed attention and soft_mask
            attn = attn_batch[i]
            soft_mask = soft_mask_batch[i]
            # Apply the current mask to the weights
            mask_f = mask.to(dtype=soft_mask.dtype)
            w = (soft_mask * attn * mask_f).unsqueeze(-1)  # [N, 1]
            context_del = torch.sum(w * emb_nodes_del, dim=0)  # [d_in]

            # Compute logits for this anchor
            rep = emb_a + emb_q + context_del
            logits = head(rep)  # [2]

            score = ce_margin_score(logits)
            
            all_scores.append(score)
        
        # Compute F1 score across all anchors for this deletion fraction
        all_scores_t = torch.stack(all_scores)  # [num_anchors]

        f1 = f1_from_logits(all_scores_t, labels_batch)
        
        f1_scores.append(f1)

    scores_t = torch.tensor(f1_scores, device=device)
    auc = torch.trapz(scores_t, fracs_t).item()
    return fracs_t, scores_t, auc


def plot_deletion_curve(fracs: List[float], scores: List[float], title: str = "Deletion curve", out_path: Optional[str] = None) -> None:
    """Utility to quickly visualize the mean deletion curve.

    fracs: percentage of top-k neighbors removed (0..1)
    scores: predicted positive probability at each deletion step
    """

    import matplotlib.pyplot as plt

    plt.plot(fracs, scores, marker="o")
    plt.xlabel("Fraction of top-k neighbors removed")
    plt.ylabel("F1 score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()


@torch.inference_mode()
def eval_epoch(
    snapshots,
    encoder: DCRNNEncoder,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    head: BinaryCEHead,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    horizon: int,
    threshold: float,
    target_channel: int,
    tau: float,
    lags: int,
    device: torch.device,
    t_start: int,
    t_end: int,
    anchors_per_snapshot: int,
    max_eval_anchors: int,
    deletion_fracs: Optional[List[float]] = None,
    compute_deletion: bool = True,
    compute_suff: bool = True,
) -> Tuple[float, dict]:
    """Evaluation with attribution metrics (deletion AUC and sufficiency)."""
    encoder.eval(); attention.eval(); query_embed.eval(); head.eval()
    ce = nn.CrossEntropyLoss()

    if deletion_fracs is None:
        deletion_fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    total_loss = 0.0
    steps = 0
    
    del_auc_list = []
    del_curves = []
    del_fracs_ref = None
    suff_list = []

    max_h = 3
    t_end_eff = min(t_end, len(snapshots) - (max_h - 1))
    h_idx = 0 if horizon == 1 else 1

    for t in range(t_start, t_end_eff):
        print(f"Eval {t} / {t_end_eff}", end="\r")
        x = _to_tensor(snapshots[t].x, device=device, dtype=torch.float)
        if horizon == 1:
            y_future = _to_tensor(snapshots[t].y, device=device, dtype=torch.float)
        else:
            y_future = _to_tensor(snapshots[t + 2].y, device=device, dtype=torch.float)

        X = x_to_batched_sequence(x, lags=lags)
        emb_nodes = encoder(X, edge_index, edge_weight)  # [N,d]
        N = emb_nodes.size(0)

        node_mask = getattr(snapshots[t], "node_mask", None)
        if node_mask is not None:
            node_mask = _to_tensor(node_mask, device=device, dtype=torch.bool).view(-1)

        if node_mask is None:
            candidates = torch.arange(N, device=device)
        else:
            candidates = torch.nonzero(node_mask, as_tuple=False).view(-1)
        if candidates.numel() == 0:
            continue

        cand = candidates
        if max_eval_anchors is not None and max_eval_anchors > 0 and cand.numel() > int(max_eval_anchors):
            perm = torch.randperm(cand.numel(), device=device)
            cand = cand[perm[:int(max_eval_anchors)]]
        if anchors_per_snapshot >= 0 and anchors_per_snapshot < cand.numel():
            perm = torch.randperm(cand.numel(), device=device)
            anchor_indices = cand[perm[:anchors_per_snapshot]]
        else:
            anchor_indices = cand

        loss_t = 0.0
        
        # Collect per-anchor data for this snapshot
        snapshot_labels = []
        snapshot_anchor_idx = []
        snapshot_attn = []
        snapshot_soft_mask = []
        snapshot_k = []

        for anchor_idx in anchor_indices.tolist():
            emb_a = emb_nodes[anchor_idx]

            emb_q = query_embed(torch.tensor(h_idx, device=device))

            context, attn, soft_mask, scores, Q, K, theta, k = attention(
                emb_q=emb_q,
                emb_a=emb_a,
                emb_nodes=emb_nodes,
                node_mask=node_mask,
                tau=tau,
                return_intermediates=True,
            )

            rep = emb_a + emb_q + context
            logits2 = head(rep)  # [2]

            # Node-level label: check if node's future value > threshold
            if y_future.dim() > 1:
                val = y_future[anchor_idx, target_channel]
            else:
                val = y_future[anchor_idx]
            y_class = (val > threshold).long()

            loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1))
            
            # Collect per-anchor data for deletion curve
            snapshot_labels.append(y_class.detach())
            snapshot_anchor_idx.append(anchor_idx)
            snapshot_attn.append(attn.detach())
            snapshot_soft_mask.append(soft_mask.detach())
            snapshot_k.append(k.detach())
            
            # Compute sufficiency for this anchor
            if compute_suff:
                suff = compute_sufficiency(
                    attention=attention,
                    head=head,
                    emb_q=emb_q,
                    emb_a=emb_a,
                    emb_nodes=emb_nodes,
                    node_mask=node_mask,
                    attn=attn,
                    soft_mask=soft_mask,
                    k_estimate=k,
                    tau=tau,
                )
                suff_list.append(suff)

        loss_t = loss_t / max(1, anchor_indices.numel())
        total_loss += float(loss_t.item())
        steps += 1
        
        # Compute deletion curve only for last eval?
        if compute_deletion and len(snapshot_labels) > 0:
            labels_batch = torch.stack(snapshot_labels) 

            fracs_t, del_scores, del_auc = compute_deletion_curve(
                encoder=encoder,
                head=head,
                labels_batch=labels_batch,
                emb_q=emb_q,
                anchor_idx=snapshot_anchor_idx,
                emb_nodes_full=emb_nodes,
                X=X,
                edge_index=edge_index,
                edge_weight=edge_weight,
                node_mask=node_mask,
                attn_batch=snapshot_attn,
                soft_mask_batch=snapshot_soft_mask,
                k_batch=snapshot_k,
                removal_fracs=deletion_fracs,
            )
            del_auc_list.append(del_auc)
            del_curves.append(del_scores.detach().cpu())
            del_fracs_ref = fracs_t.detach().cpu()

    avg_loss = total_loss / max(1, steps)

    metrics = {
        "deletion_auc": float(sum(del_auc_list) / max(1, len(del_auc_list))),
        "deletion_curve": torch.stack(del_curves).mean(dim=0).tolist() if del_curves else [],
        "deletion_fracs": del_fracs_ref.tolist() if del_fracs_ref is not None else [],
        "sufficiency": float(sum(suff_list) / max(1, len(suff_list))),
    }
    return avg_loss, metrics


def train_one_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_pemsbay_from_npy(ADJ_PATH, VALUES_PATH, lags=LAGS, horizon=1, adj_threshold=ADJ_THRESHOLD)
    snapshots = list(dataset)
    snapshots = snapshots[:int(TOTAL_RATIO * len(snapshots))]

    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long, device=device)
    edge_weight = torch.tensor(dataset.edge_weight, dtype=torch.float, device=device)

    x0 = torch.tensor(snapshots[0].x)
    assert x0.dim() == 2 and x0.shape[1] % LAGS == 0
    num_features = int(x0.shape[1] // LAGS)

    encoder = DCRNNEncoder(in_channels=num_features, d_emb=D_EMB, K=DCRNN_K).to(device)
    query_embed = nn.Embedding(2, D_EMB).to(device)

    attention = QKOnlySoftTopKAttention(
        d_in=D_EMB,
        d_out=D_EMB,
        tau=TAU,
        init_k_frac=INIT_K_FRAC,
        k_min=K_MIN,
        k_max=K_MAX,
        normalize_qk=False,
        newton_iters=15,
        newton_damping=1.0,
    ).to(device)

    head = BinaryCEHead(D_EMB).to(device)

    metric_loss = MetricLoss(num_classes=2, d_emb=D_EMB, code_size=D_EMB, device=device).to(device)
    mixup = MixupWithMemory(num_classes=2, d_emb=D_EMB, device=device)

    params = (
        list(encoder.parameters())
        + list(query_embed.parameters())
        + list(attention.parameters())
        + list(head.parameters())
        + list(metric_loss.parameters())
    )
    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss()

    T = len(snapshots)
    train_end = max(1, int(TRAIN_RATIO * T))
    test_start = train_end
    test_end = T

    h_idx = 0 if HORIZON == 1 else 1

    print(f"=== ({MODEL_NAME}) node-level CE with attribution: HORIZON={HORIZON} ===")
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end/T:.1%}), test [{test_start}, {test_end})")

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); attention.train(); query_embed.train(); head.train(); metric_loss.train()

        total = 0.0
        steps = 0

        max_h = 3
        t_end_eff = min(train_end, len(snapshots) - (max_h - 1))

        for t in range(0, t_end_eff):
            print(f"{t} / {T}", end="\r")
            x = _to_tensor(snapshots[t].x, device=device, dtype=torch.float)
            if HORIZON == 1:
                y_future = _to_tensor(snapshots[t].y, device=device, dtype=torch.float)
            else:
                y_future = _to_tensor(snapshots[t + 2].y, device=device, dtype=torch.float)

            X = x_to_batched_sequence(x, lags=LAGS)
            emb_nodes = encoder(X, edge_index, edge_weight)  # [N,d]
            N = emb_nodes.size(0)

            node_mask = getattr(snapshots[t], "node_mask", None)
            if node_mask is not None:
                node_mask = _to_tensor(node_mask, device=device, dtype=torch.bool).view(-1)

            if node_mask is None:
                candidates = torch.arange(N, device=device)
            else:
                candidates = torch.nonzero(node_mask, as_tuple=False).view(-1)
            if candidates.numel() == 0:
                continue

            if ANCHORS_PER_SNAPSHOT_TRAIN < 0 or ANCHORS_PER_SNAPSHOT_TRAIN >= candidates.numel():
                anchor_indices = candidates
            else:
                perm = torch.randperm(candidates.numel(), device=device)
                anchor_indices = candidates[perm[:ANCHORS_PER_SNAPSHOT_TRAIN]]

            loss_t = 0.0

            all_context = []
            all_query = []
            all_target = []

            for anchor_idx in anchor_indices.tolist():
                emb_a = emb_nodes[anchor_idx]

                emb_q = query_embed(torch.tensor(h_idx, device=device))

                context, attn, soft_mask, scores, Q, K, theta, k = attention(
                    emb_q=emb_q,
                    emb_a=emb_a,
                    emb_nodes=emb_nodes,
                    node_mask=node_mask,
                    tau=TAU,
                    return_intermediates=True,
                )

                rep = emb_a + emb_q + context
                logits2 = head(rep)

                # Node-level label: check if node's future value > threshold
                if y_future.dim() > 1:
                    val = y_future[anchor_idx, TARGET_CHANNEL]
                else:
                    val = y_future[anchor_idx]
                y_class = (val > THRESHOLD).long()

                loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1))

                k_hard = int(round(float(k.detach().cpu().item())))
                k_hard = max(2, k_hard)
                loss_t = loss_t + BETA_DIV * diversity_loss(attn, emb_nodes, k=k_hard, node_mask=node_mask)

                all_context.append(context)
                all_query.append(emb_q)
                all_target.append(y_class)

            loss_t = loss_t / max(1, anchor_indices.numel())

            if len(all_context) > 0:
                ctx = torch.stack(all_context)
                qry = torch.stack(all_query)
                tgt = torch.stack(all_target).long()

                mix_ctx, mix_tgt, tgt_i, tgt_j, mix_qry = mixup.get_mixup_samples(ctx, qry, tgt)

                mloss = metric_loss(ctx, mix_ctx, qry, mix_qry, tgt, tgt_i, tgt_j)
                if torch.is_tensor(mloss):
                    mloss = mloss.mean()
                loss_t = loss_t + BETA_METRIC * mloss

                mix_rep = mix_ctx + mix_qry
                mix_logits2 = head(mix_rep)  # [B,2]
                mix_ce = F.cross_entropy(mix_logits2, mix_tgt, reduction="mean")
                loss_t = loss_t + BETA_MIXUP * mix_ce

            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            opt.step()

            total += float(loss_t.detach().cpu().item())
            steps += 1

        train_loss = total / max(1, steps)

        k_frac = float(torch.sigmoid(attention.k_logit).detach().cpu())
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"k_frac={k_frac:.4f}"
        )

        with torch.inference_mode():
            test_loss, test_m = eval_epoch(
                snapshots, encoder, attention, query_embed, head,
                edge_index, edge_weight,
                HORIZON, THRESHOLD, TARGET_CHANNEL, TAU, LAGS, device,
                t_start=test_start, t_end=test_end,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_EVAL,
                max_eval_anchors=MAX_EVAL_ANCHORS,
                deletion_fracs=DELETION_FRACS,
                compute_deletion=False,
                compute_suff=True,
            )

    print(
        f"test_loss={test_loss:.6f} "
        f"del_AUC={test_m['deletion_auc']:.3f}, suff={test_m['sufficiency']:.3f} "
    )

    # Plot deletion curves for last epoch test set
    if len(test_m["deletion_fracs"]) > 0 and len(test_m["deletion_curve"]) > 0:
        plot_deletion_curve(
            fracs=test_m["deletion_fracs"],
            scores=test_m["deletion_curve"],
            title=f"Epoch {EPOCHS} h{HORIZON} Deletion Curve",
            out_path=f"deletion_curve_h{HORIZON}_no_div.png",
        )
        np.savez(
            f"deletion_curve_h{HORIZON}_no_div.npz",
            fracs=np.array(test_m["deletion_fracs"]),
            scores=np.array(test_m["deletion_curve"]),
            auc=test_m["deletion_auc"],
        )
        print(f"Saved deletion curve data to deletion_curve_h{HORIZON}_no_div.npz")


if __name__ == "__main__":
    train_one_run()
