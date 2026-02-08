#   Subgraph label is based on RANGE (max - min) within k-hop subgraph,
#   not mean.

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from pems_bay import load_pemsbay_from_npy
#from attention_layer_alt import QKOnlySoftTopKAttention
from attention_layer import QKOnlySoftTopKAttention
from diversity_loss_f import diversity_loss
from mixup import MixupWithMemory
from metric_loss import MetricLoss
from metrics_bin_updated import (
    binary_auc_from_logits,
    roc_auc_from_logits,
    ap_from_logits,
    precision_at_k_from_logits,
    recall_at_k_from_logits,
    f1_from_logits,
    ndcg_at_k_from_logits,
)

# =========================
# MODEL CHOICE
# =========================
# Choose one of: "DCRNN", "SEHTGNN", "TASER"
MODEL_NAME = "DCRNN"

# TASER adapter settings (taser_encoder_pyg.py)
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

K_HOP = 2 #2
THRESHOLD = 7.0
TARGET_CHANNEL = 0
ADJ_THRESHOLD = 0.0

EPOCHS = 5 
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# attention, mixup, metric loss settings
INIT_K_FRAC = 0.001
K_MIN = 0.001
K_MAX = 0.01
k_abs_min = 5
k_abs_max = 100
TAU = 0.05
BETA_DIV = 0.1
BETA_METRIC = 0.1
BETA_MIXUP = 1.0

# batching (to reduce memory)
ANCHORS_PER_SNAPSHOT_TRAIN = 256
ANCHORS_PER_SNAPSHOT_EVAL = 256
MAX_EVAL_ANCHORS = 512

# split
TRAIN_RATIO = 0.2  # 20% train, 80% test
TOTAL_RATIO = 0.1 # total % data for train+test; 0.2 is around 10k snapshots, nearly 1 month
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


def precompute_khop_subsets(edge_index: torch.Tensor, num_nodes: int, k_hop: int) -> List[torch.Tensor]:
    edge_index_cpu = edge_index.detach().cpu()
    subsets = []
    for i in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=i,
            num_hops=k_hop,
            edge_index=edge_index_cpu,
            relabel_nodes=False,
            num_nodes=num_nodes,
        )
        subsets.append(subset.to(dtype=torch.long, device=torch.device("cpu")))
    return subsets


def subgraph_label_class_range(
    y_future: torch.Tensor,
    subset: torch.Tensor,
    threshold: float,
    target_channel: int,
) -> torch.Tensor:
    # Task: range over subgraph S is max(y_j) - min(y_j).
    # Label 1 if range > threshold else 0.    
    if y_future.dim() > 1:
        vals = y_future[subset, target_channel]
    else:
        vals = y_future[subset]

    vmax = vals.max()
    vmin = vals.min()
    rng = vmax - vmin
    #print('current range:{}\ttotal nodes:{}'.format(rng,len(subset)))
    
    return (rng > threshold).long()


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
            from SEHTGNN import SEHTGNN as _SEHTGNN  # fallback
            self.cell = _SEHTGNN(in_channels=in_channels, out_channels=d_emb, K=K, time_window=LAGS)                

        elif name == "TASER":
            from taser import TaserTGNNCell as _TASER
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


def ce_margin_score(logits2: torch.Tensor) -> torch.Tensor:
    return logits2[..., 1] - logits2[..., 0]


@torch.inference_mode()
def eval_epoch(
    snapshots,
    encoder: DCRNNEncoder,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,
    head: BinaryCEHead,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    khop_subsets: List[torch.Tensor],
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
    k_eval: int,
    f1_thr: float,
) -> Tuple[float, dict]:
    encoder.eval(); attention.eval(); query_embed.eval(); head.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    steps = 0

    all_scores_cpu = []
    all_labels_cpu = []
    p_at_k_list = []

    max_h = 3
    t_end_eff = min(t_end, len(snapshots) - (max_h - 1))
    h_idx = 0 if horizon == 1 else 1

    for t in range(t_start, t_end_eff):
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

        snapshot_scores = torch.full((N,), float("-inf"), device=device)
        snapshot_labels = torch.zeros((N,), device=device)

        loss_t = 0.0

        for anchor_idx in anchor_indices.tolist():
            emb_a = emb_nodes[anchor_idx]
            subset = khop_subsets[anchor_idx].to(device=device)

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

            # NEW: range-based label
            y_class = subgraph_label_class_range(y_future, subset, threshold, target_channel)

            loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1))

            score = ce_margin_score(logits2).detach()
            snapshot_scores[anchor_idx] = score
            snapshot_labels[anchor_idx] = y_class.detach().float()
            all_scores_cpu.append(score.cpu())
            all_labels_cpu.append(y_class.detach().cpu().float())

        loss_t = loss_t / max(1, anchor_indices.numel())
        total_loss += float(loss_t.item())
        steps += 1
        
        p_at_k_list.append(precision_at_k_from_logits(snapshot_scores, snapshot_labels, k=k_eval))

    avg_loss = total_loss / max(1, steps)
    if steps == 0 or len(all_scores_cpu) == 0:
        return avg_loss, {"auc": 0.5, "f1": 0.0, "p@k": 0.0}

    scores_cpu = torch.stack(all_scores_cpu).view(-1)
    labels_cpu = torch.stack(all_labels_cpu).view(-1)

    metrics = {
        "auc": binary_auc_from_logits(scores_cpu, labels_cpu),
        "roc-auc": roc_auc_from_logits(scores_cpu, labels_cpu),
        "ap": ap_from_logits(scores_cpu, labels_cpu),
        "f1": f1_from_logits(scores_cpu, labels_cpu),
        #"p@k": float(sum(p_at_k_list) / max(1, len(p_at_k_list))),
        "p@k": precision_at_k_from_logits(scores_cpu, labels_cpu, k=K_EVAL),
        "r@k": recall_at_k_from_logits(scores_cpu, labels_cpu, k=K_EVAL),
        "ndcg@k": ndcg_at_k_from_logits(scores_cpu, labels_cpu, k=K_EVAL),
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
    num_nodes = int(x0.shape[0])

    khop_subsets = precompute_khop_subsets(edge_index, num_nodes=num_nodes, k_hop=K_HOP)

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
        k_abs_min=k_abs_min,
        k_abs_max=k_abs_max,
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
    train_end = max(1, int(TRAIN_RATIO * T))  # 10% train
    test_start = train_end
    test_end = T

    h_idx = 0 if HORIZON == 1 else 1

    print(f"=== ({MODEL_NAME}) single-horizon CE: HORIZON={HORIZON} k-hop={K_HOP} ===")
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end/T:.1%}), test [{test_start}, {test_end})")

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); attention.train(); query_embed.train(); head.train(); metric_loss.train()

        total = 0.0
        steps = 0

        max_h = 3
        t_end_eff = min(train_end, len(snapshots) - (max_h - 1))

        for t in range(0, t_end_eff):
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
                subset = khop_subsets[anchor_idx].to(device=device)

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

                # NEW: range-based label
                y_class = subgraph_label_class_range(y_future, subset, THRESHOLD, TARGET_CHANNEL)

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

        with torch.inference_mode():
            test_loss, test_m = eval_epoch(
                snapshots, encoder, attention, query_embed, head,
                edge_index, edge_weight, khop_subsets,
                HORIZON, THRESHOLD, TARGET_CHANNEL, TAU, LAGS, device,
                t_start=test_start, t_end=test_end,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_EVAL,
                max_eval_anchors=MAX_EVAL_ANCHORS,
                k_eval=K_EVAL, f1_thr=F1_THR
            )

        k_frac = float(torch.sigmoid(attention.k_logit).detach().cpu())
        extra = ""
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} (AUC={test_m['auc']:.3f},  ROC-AUC={test_m['roc-auc']:.3f}, F1={test_m['f1']:.3f}, P@{K_EVAL}={test_m['p@k']:.3f}, R@{K_EVAL}={test_m['r@k']:.3f}, NDCG@{K_EVAL}={test_m['ndcg@k']:.3f})"
            + extra
        )


if __name__ == "__main__":
    train_one_run()

