# Subgraph label is based on RANGE (max - min) within k-hop subgraph

from typing import Optional, Tuple, List
import time  # for timing train/eval
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch_geometric.utils import k_hop_subgraph
from soft_topk_attn.data.pems_bay import load_pemsbay_from_npy
from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.faiss_gpu_retriever import FaissGpuRetriever  # FAISS GPU candidate retriever (faiss-gpu==1.7.2)
from soft_topk_attn.models.diversity_loss_f import diversity_loss
from soft_topk_attn.models.mixup import MixupWithMemory
from soft_topk_attn.models.metric_loss import MetricLoss
from soft_topk_attn.models.metrics_bin import (
    binary_auc_from_logits,
    binary_f1_from_logits,
    precision_at_k_from_logits,
)


# =========================
# timing helpers
# =========================
def _sync_if_cuda(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _now() -> float:
    return time.perf_counter()


# =========================
# MODEL CHOICE
# =========================
# Choose one of: "DCRNN", "SEHTGNN", "TASER"
MODEL_NAME = "TASER"

# If False: direct baseline
# If True : full model
USE_ATTN = True
# Only relevant when USE_ATTN=True
# If False: disable mixup and metric-loss parts.
USE_MIXUP = True

# TASER adapter settings (taser_encoder_pyg.py)
TASER_NUM_NEIGHBORS = 16
TASER_DIM_TIME = 16
TASER_ATT_HEAD = 4

# =========================
# SETTINGS
# =========================
HORIZON = 1  # set to 1 or 3 (predict 1-step or 3-step future)
LAGS = 12
D_EMB = 32
DCRNN_K = 2

K_HOP = 2  # 2
THRESHOLD = 7.0
TARGET_CHANNEL = 0
ADJ_THRESHOLD = 0.0

EPOCHS = 10
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# attention / aux losses
INIT_K_FRAC = 0.01
K_MIN = 0.01
K_MAX = 0.1
K_ABS_MIN = 5
K_ABS_MAX = 10
TAU_MAX = 0.8
TAU_MIN = 0.1
BETA = 3
BETA_DIV = 0.1
BETA_METRIC = 0.5
ALPHA_0 = 1.0
ALPHA_MIN = 0.1
LAMBDA_0 = 1.0
LAMBDA_MIN = 0.1

# batching (to reduce memory)
ANCHORS_PER_SNAPSHOT_TRAIN = -1
ANCHORS_PER_SNAPSHOT_EVAL = -1
MAX_EVAL_ANCHORS = -1

# split
TRAIN_RATIO = 0.1  # 10% train, 90% test
TOTAL_RATIO = 0.03 # 0.03  # total % data for train+test
ADJ_PATH = "data/pems_adj_mat.npy"
VALUES_PATH = "data/pems_node_values.npy"

USE_FAISS_GPU = True
FAISS_TOPM = K_ABS_MAX*5  # m candidates per anchor (m >> k)
FAISS_UPDATE_EVERY = 3  # rebuild index every R snapshots (1 = every snapshot)
FAISS_UNION_KHOP = True  # union ANN candidates with k-hop subset (recommended)
FAISS_MAX_CAND = 2048  # cap candidate set size after union
FAISS_METRIC = 'ip'  # 'ip' (inner product) or 'l2'

# set True to hard-fail if torch-GPU fastpath is not available
FAISS_REQUIRE_TORCH_GPU = True
# =========================


def _to_tensor(x, device, dtype=torch.float):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


# =========================
# Avoids the expensive unique/sort on GPU for every anchor.
# =========================
def _union_faiss_and_khop(
        cand_ids: torch.Tensor,
        subset: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        num_nodes: int,
) -> torch.Tensor:
    # cand_ids: [m] long (CUDA)
    # subset : [s] long (CUDA)
    if cand_ids.numel() == 0:
        cand_ids = subset
    if cand_ids.numel() == 0:
        return cand_ids

    # filter subset by node_mask if provided
    subset_u = subset
    if node_mask is not None and subset_u.numel() > 0:
        subset_u = subset_u[node_mask[subset_u]]

    if subset_u.numel() == 0:
        return cand_ids

    # bool "seen" mask: O(N) ; cheap compared to torch.unique/sort
    seen = torch.zeros((num_nodes,), device=cand_ids.device, dtype=torch.bool)
    seen[cand_ids] = True
    add = subset_u[~seen[subset_u]]
    if add.numel() > 0:
        cand_ids = torch.cat([cand_ids, add], dim=0)
    return cand_ids

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
    """
    Task: range over subgraph S is max(y_j) - min(y_j).
    Label 1 if range > threshold else 0.
    """
    if y_future.dim() > 1:
        vals = y_future[subset, target_channel]
    else:
        vals = y_future[subset]
    vmax = vals.max()
    vmin = vals.min()
    rng = vmax - vmin
    # print('current range:{}\ttotal nodes:{}'.format(rng,len(subset)))
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
            from soft_topk_attn.models.SEHTGNN import SEHTGNN as _SEHTGNN  # fallback
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


class SubgraphHead(nn.Module):
    """Pooled subgraph representation -> logits."""

    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, sub_rep: torch.Tensor) -> torch.Tensor:
        return self.lin(sub_rep)


def pool_subgraph_mean(emb_nodes: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """Mean pooling over k-hop subset."""
    return emb_nodes[subset].mean(dim=0)


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
        head: nn.Module,
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
        # full-model components (only used if USE_ATTN=True)
        attention: Optional[QKOnlySoftTopKAttention] = None,
        query_embed: Optional[nn.Embedding] = None,
        retriever: Optional[FaissGpuRetriever] = None,
        use_faiss_gpu: Optional[bool] = None,
) -> Tuple[float, dict]:
    """
    If USE_ATTN:
        rep = emb_a + emb_q + context
        logits = head(rep)   (head is BinaryCEHead)
    Else:
        sub_rep = mean(emb_nodes[subset])
        logits  = head(sub_rep) (head is SubgraphHead)
    """
    encoder.eval()
    head.eval()
    if USE_ATTN:
        assert attention is not None and query_embed is not None
        attention.eval()
        query_embed.eval()

    _use_faiss = USE_FAISS_GPU if use_faiss_gpu is None else bool(use_faiss_gpu)
    if USE_ATTN and _use_faiss:
        assert retriever is not None  # FAISS retriever must be provided

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

        # =========================
        # Build/update FAISS GPU index once per FAISS_UPDATE_EVERY snapshot
        # Key space: K = W_k(emb_nodes). Index only valid nodes (node_mask if provided).
        # =========================
        if USE_ATTN and _use_faiss:
            if node_mask is None:
                faiss_valid_idx = torch.arange(N, device=device, dtype=torch.long)
            else:
                faiss_valid_idx = torch.nonzero(node_mask, as_tuple=False).view(-1).long()
            # numel(): return the number of elements in a tensor.
            if faiss_valid_idx.numel() > 0 and (t % int(FAISS_UPDATE_EVERY) == 0):
                K_valid = attention.W_k(emb_nodes[faiss_valid_idx]).detach()
                retriever.build(K_valid, valid_idx=faiss_valid_idx, normalize=bool(attention.normalize_qk),
                                require_torch_gpu=FAISS_REQUIRE_TORCH_GPU)

        # =========================
        # Batch FAISS search for all anchors in this snapshot (one call, not per-anchor)
        # =========================
        cand_batch = None
        if USE_ATTN and _use_faiss:
            emb_q = query_embed(torch.tensor(h_idx, device=device))
            emb_a_batch = emb_nodes[anchor_indices]  # [A,d]
            qa_batch = emb_a_batch + emb_q  # broadcast emb_q
            Q_batch = attention.W_q(qa_batch)
            if attention.normalize_qk:
                Q_batch = F.normalize(Q_batch, dim=-1)
            cand_batch, _ = retriever.search(
                Q_batch,
                topm=FAISS_TOPM,
                normalize=bool(attention.normalize_qk),
                require_torch_gpu=FAISS_REQUIRE_TORCH_GPU,
            )

        for _i, anchor_idx in enumerate(anchor_indices.tolist()):  # _i indexes cand_batch
            subset = khop_subsets[anchor_idx].to(device=device)

            if USE_ATTN:
                emb_a = emb_nodes[anchor_idx]
                emb_q = query_embed(torch.tensor(h_idx, device=device))
                # =========================
                # Use FAISS candidates to run attention on a smaller candidate set
                # =========================
                if _use_faiss:
                    # Use precomputed FAISS candidates for this anchor (avoid per-anchor retriever.search)

                    cand_ids = cand_batch[_i].view(-1)

                    if FAISS_UNION_KHOP:
                        cand_ids = _union_faiss_and_khop(cand_ids, subset, node_mask, N)

                    if cand_ids.numel() > FAISS_MAX_CAND:
                        cand_ids = cand_ids[:FAISS_MAX_CAND]

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
                        node_mask=node_mask,
                        tau=tau,
                        return_intermediates=True,
                    )
                rep = emb_a + emb_q + context
                logits2 = head(rep)  # [2]
            else:
                sub_rep = pool_subgraph_mean(emb_nodes, subset)  # [d]
                logits2 = head(sub_rep)  # [2]

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
        "f1": binary_f1_from_logits(scores_cpu, labels_cpu, thr=f1_thr),
        "p@k": float(sum(p_at_k_list) / max(1, len(p_at_k_list))),
    }
    return avg_loss, metrics


def train_one_run(use_faiss_gpu: Optional[bool] = None):  # CHANGED: override for timing comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # local flag (default to global USE_FAISS_GPU)
    _use_faiss = USE_FAISS_GPU if use_faiss_gpu is None else bool(use_faiss_gpu)

    # timing stats
    stats = {
        'use_faiss': _use_faiss,
        'epoch_train_time': [],
        'epoch_eval_time': [],
        'total_train_time': 0.0,
        'total_eval_time': 0.0,
    }

    dataset = load_pemsbay_from_npy(ADJ_PATH, VALUES_PATH, lags=LAGS, horizon=1, adj_threshold=ADJ_THRESHOLD)
    snapshots = list(dataset)
    snapshots = snapshots[:int(TOTAL_RATIO * len(snapshots))]

    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long, device=device)
    edge_weight = torch.tensor(dataset.edge_weight, dtype=torch.float, device=device)

    x0 = torch.tensor(snapshots[0].x)
    assert x0.dim() == 2 and x0.shape[1] % LAGS == 0
    num_features = int(x0.shape[1] // LAGS)
    num_nodes = int(x0.shape[0])

    # Precompute k-hop subsets once
    khop_subsets = precompute_khop_subsets(edge_index, num_nodes=num_nodes, k_hop=K_HOP)

    encoder = DCRNNEncoder(in_channels=num_features, d_emb=D_EMB, K=DCRNN_K).to(device)

    # Heads / optional components
    query_embed = None
    attention = None
    metric_loss = None
    mixup = None
    retriever = None  # FAISS retriever

    if USE_ATTN:
        query_embed = nn.Embedding(2, D_EMB).to(device) # Use LLM embeddings instead (future)
        attention = QKOnlySoftTopKAttention(
            d_in=D_EMB,
            d_out=D_EMB,
            tau=TAU_MIN,
            init_k_frac=INIT_K_FRAC,
            k_min=K_MIN,
            k_max=K_MAX,
            normalize_qk=False,
            newton_iters=15,
            newton_damping=1.0,
            # learn absolute k in [K_ABS_MIN, K_ABS_MAX]
            k_abs_min=K_ABS_MIN,
            k_abs_max=K_ABS_MAX,
        ).to(device)
        head = BinaryCEHead(D_EMB).to(device)

        if USE_MIXUP:
            metric_loss = MetricLoss(num_classes=2, d_emb=D_EMB, code_size=D_EMB, device=device).to(device)
            mixup = MixupWithMemory(num_classes=2, d_emb=D_EMB, device=device)

        # Init FAISS GPU retriever (built once per snapshot)
        if _use_faiss:
            retriever = FaissGpuRetriever(device=device, metric=FAISS_METRIC)

        params = (
                list(encoder.parameters())
                + list(query_embed.parameters())
                + list(attention.parameters())
                + list(head.parameters())
        )
        if USE_MIXUP:
            params = params + list(metric_loss.parameters())
    else:
        head = SubgraphHead(D_EMB).to(device)
        params = list(encoder.parameters()) + list(head.parameters())

    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss()

    T = len(snapshots)
    train_end = max(1, int(TRAIN_RATIO * T))
    test_start = train_end
    test_end = T

    h_idx = 0 if HORIZON == 1 else 1

    if not USE_ATTN:
        mode_str = "DIRECT"
    else:
        parts = ["ATN"]
        if USE_MIXUP:
            parts.append("MIX+MET")
        mode_str = "+".join(parts)
    print(f"=== {mode_str} ({MODEL_NAME}) subgraph-range: HORIZON={HORIZON} k-hop={K_HOP} ===")
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end / T:.1%}), test [{test_start}, {test_end})")
    print(f"[Timing] FAISS={'ON' if _use_faiss else 'OFF'}")

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        head.train()
        if USE_ATTN:
            attention.train()
            query_embed.train()
            if USE_MIXUP:
                metric_loss.train()

        # Anneal TAU
        tau_epoch = TAU_MAX * math.exp(-BETA * (epoch - 1)) + TAU_MIN 

        # Anneal metric lambda
        beta_metric_epoch = LAMBDA_0 * math.exp(-BETA * (epoch - 1)) + LAMBDA_MIN

        # Anneal a(t) for mixup lambda
        alpha_epoch = ALPHA_0 * math.exp(-BETA * (epoch - 1)) + ALPHA_MIN

        beta_dist = Beta(alpha_epoch, alpha_epoch)

        beta_mixup_epoch = beta_dist.sample()

        metric_loss.lamb = beta_mixup_epoch

        # Start timing training for this epoch
        _sync_if_cuda(device)
        _t_train0 = _now()

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

            # =========================
            # Build/update FAISS GPU index once per snapshot
            # =========================
            if USE_ATTN and _use_faiss:
                assert retriever is not None
                if node_mask is None:
                    faiss_valid_idx = torch.arange(N, device=device, dtype=torch.long)
                else:
                    faiss_valid_idx = torch.nonzero(node_mask, as_tuple=False).view(-1).long()
                if faiss_valid_idx.numel() > 0 and (t % int(FAISS_UPDATE_EVERY) == 0):
                    K_valid = attention.W_k(emb_nodes[faiss_valid_idx]).detach()
                    retriever.build(K_valid, valid_idx=faiss_valid_idx, normalize=bool(attention.normalize_qk),
                                    require_torch_gpu=FAISS_REQUIRE_TORCH_GPU)

            # =========================
            # Batch FAISS search for all anchors in this snapshot (one call, not per-anchor)
            # =========================
            cand_batch = None
            if USE_ATTN and _use_faiss:
                emb_q = query_embed(torch.tensor(h_idx, device=device))
                emb_a_batch = emb_nodes[anchor_indices]  # [A,d]
                qa_batch = emb_a_batch + emb_q
                Q_batch = attention.W_q(qa_batch)
                if attention.normalize_qk:
                    Q_batch = F.normalize(Q_batch, dim=-1)
                cand_batch, _ = retriever.search(
                    Q_batch,
                    topm=FAISS_TOPM,
                    normalize=bool(attention.normalize_qk),
                    require_torch_gpu=FAISS_REQUIRE_TORCH_GPU,
                )

            all_context = []
            all_query = []
            all_target = []

            for _i, anchor_idx in enumerate(anchor_indices.tolist()):  # _i indexes cand_batch
                subset = khop_subsets[anchor_idx].to(device=device)
                y_class = subgraph_label_class_range(y_future, subset, THRESHOLD, TARGET_CHANNEL)

                if USE_ATTN:
                    emb_a = emb_nodes[anchor_idx]
                    emb_q = query_embed(torch.tensor(h_idx, device=device))

                    # =========================
                    # Use FAISS candidates to run attention on a smaller candidate set
                    # =========================
                    if _use_faiss:
                        # Use precomputed FAISS candidates for this anchor (avoid per-anchor retriever.search)

                        cand_ids = cand_batch[_i].view(-1)

                        if FAISS_UNION_KHOP:
                                cand_ids = _union_faiss_and_khop(cand_ids, subset, node_mask, N)  # CHANGED (2)

                        if cand_ids.numel() > FAISS_MAX_CAND:
                            cand_ids = cand_ids[:FAISS_MAX_CAND]

                        emb_nodes_c = emb_nodes[cand_ids]
                        context, attn, soft_mask, scores, Q, K, theta, k = attention.forward_candidates(
                            emb_q=emb_q,
                            emb_a=emb_a,
                            emb_nodes_cand=emb_nodes_c,
                            node_mask=None,
                            tau=tau_epoch,
                            return_intermediates=True,
                        )
                        _div_attn = attn
                        _div_nodes = emb_nodes_c
                        _div_mask = None
                    else:
                        context, attn, soft_mask, scores, Q, K, theta, k = attention(
                            emb_q=emb_q,
                            emb_a=emb_a,
                            emb_nodes=emb_nodes,
                            node_mask=node_mask,
                            tau=tau_epoch,
                            return_intermediates=True,
                        )
                        _div_attn = attn
                        _div_nodes = emb_nodes
                        _div_mask = node_mask

                    rep = emb_a + emb_q + context
                    logits2 = head(rep)
                    loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1))

                    # Diversity loss uses hard k derived from solved k
                    k_hard = int(round(float(k.detach().cpu().item())))
                    k_hard = max(2, k_hard)
                    loss_t = loss_t + BETA_DIV * diversity_loss(_div_attn, _div_nodes, k=k_hard,
                                                                node_mask=_div_mask)

                    all_context.append(context)
                    all_query.append(emb_q)
                    all_target.append(y_class)
                else:
                    sub_rep = pool_subgraph_mean(emb_nodes, subset)
                    logits2 = head(sub_rep)
                    loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1))

            loss_t = loss_t / max(1, anchor_indices.numel())

            if USE_ATTN and USE_MIXUP and len(all_context) > 0:
                ctx = torch.stack(all_context)
                qry = torch.stack(all_query)
                tgt = torch.stack(all_target).long()

                if mixup is None or metric_loss is None:
                    raise RuntimeError("USE_MIXUP is True but mixup/metric_loss is not initialized.")

                mix_ctx, mix_tgt, tgt_i, tgt_j, mix_qry = mixup.get_mixup_samples(ctx, qry, tgt)

                # Metric loss (always on when USE_MIXUP=True)
                mloss = metric_loss(ctx, mix_ctx, qry, mix_qry, tgt, tgt_i, tgt_j)
                if torch.is_tensor(mloss):
                    mloss = mloss.mean()
                loss_t = loss_t + beta_metric_epoch * mloss  # annealed

                # Mixup classification loss
                mix_rep = mix_ctx + mix_qry
                mix_logits2 = head(mix_rep)  # [B,2]
                mix_ce = F.cross_entropy(mix_logits2, mix_tgt, reduction="mean")
                loss_t = loss_t + mix_ce  # annealed

            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            opt.step()

            total += float(loss_t.detach().cpu().item())
            steps += 1

        train_loss = total / max(1, steps)

        # end timing training for this epoch
        _sync_if_cuda(device)
        _t_train1 = _now()
        _epoch_train_time = _t_train1 - _t_train0
        stats['epoch_train_time'].append(_epoch_train_time)
        stats['total_train_time'] += _epoch_train_time

        # start timing evaluation for this epoch
        _sync_if_cuda(device)
        _t_eval0 = _now()
        with torch.inference_mode():
            test_loss, test_m = eval_epoch(
                snapshots, encoder, head,
                edge_index, edge_weight, khop_subsets,
                HORIZON, THRESHOLD, TARGET_CHANNEL, tau_epoch, LAGS, device,
                t_start=test_start, t_end=test_end,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_EVAL,
                max_eval_anchors=MAX_EVAL_ANCHORS,
                k_eval=K_EVAL, f1_thr=F1_THR,
                attention=attention,
                query_embed=query_embed,
                retriever=retriever,  # CHANGED
                use_faiss_gpu=_use_faiss,  # CHANGED
            )
        # end timing evaluation for this epoch
        _sync_if_cuda(device)
        _t_eval1 = _now()
        _epoch_eval_time = _t_eval1 - _t_eval0
        stats['epoch_eval_time'].append(_epoch_eval_time)
        stats['total_eval_time'] += _epoch_eval_time

        if USE_ATTN:
            u = float(torch.sigmoid(attention.k_logit).detach().cpu())

            # if model use absolute-k bounds:
            if attention.k_abs_min is not None:
                k_abs = attention.k_abs_min + (attention.k_abs_max - attention.k_abs_min) * u
                extra = f" u={u:.4f} k_abs={k_abs:.2f}"
            else:
                k_frac = attention.k_min + (attention.k_max - attention.k_min) * u
                extra = f" u={u:.4f} k_frac={k_frac:.4f}"

        else:
            extra = ""

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} (AUC={test_m['auc']:.3f}, F1={test_m['f1']:.3f}, P@{K_EVAL}={test_m['p@k']:.3f})"
            + extra
            + f"  tau={tau_epoch:.4f} beta_metric={beta_metric_epoch:.4g} beta_mixup={beta_mixup_epoch:.4g}  time_train={_epoch_train_time:.2f}s  time_eval={_epoch_eval_time:.2f}s"  # CHANGED
        )

    # return timing stats for comparison
    return stats


if __name__ == "__main__":
    # compare timing with/without FAISS in the same script run
    print("\n===== Timing comparison: FAISS OFF =====")
    stats_no = train_one_run(use_faiss_gpu=False)
    print("\n===== Timing comparison: FAISS ON =====")
    stats_yes = train_one_run(use_faiss_gpu=True)


    def _summ(s):
        et = s['epoch_train_time']
        ee = s['epoch_eval_time']
        return {
            'use_faiss': s['use_faiss'],
            'total_train_time_s': s['total_train_time'],
            'total_eval_time_s': s['total_eval_time'],
            'avg_train_time_per_epoch_s': sum(et) / max(1, len(et)),
            'avg_eval_time_per_epoch_s': sum(ee) / max(1, len(ee)),
        }


    print("\n===== Summary =====")
    print("No-FAISS:", _summ(stats_no))
    print("FAISS   :", _summ(stats_yes))
