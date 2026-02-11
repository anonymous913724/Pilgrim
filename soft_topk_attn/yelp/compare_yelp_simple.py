# ============================================================
# Converting Yelp HeteroData snapshots into PEMS-like inputs:
#   - Nodes: businesses
#   - Edges: ('business','geo','business') static graph
#   - Node features: aggregate review info and concatenate with business.x
#   - Time window: lags months, flattened to [Nb, lags*F]
# ============================================================

from typing import Optional, Tuple, List, Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.utils import k_hop_subgraph

from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.faiss_gpu_retriever import FaissGpuRetriever  # faiss-gpu==1.7.2
from soft_topk_attn.models.diversity_loss_f import diversity_loss
#from mixup import MixupWithMemory
from soft_topk_attn.models.mixup_cap import MixupWithMemory
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now() -> float:
    return time.perf_counter()


def _to_tensor(x, device, dtype=torch.float):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


# =========================
# MODEL CHOICE
# =========================
# Choose one of: "DCRNN", "SEHTGNN", "TASER"
MODEL_NAME = "DCRNN"

# If False: direct baseline (mean pool on k-hop subset)
# If True : full model (attention + FAISS + optional mixup/metric)
USE_ATTN = True

# Only relevant when USE_ATTN=True
# If False: disable mixup and metric-loss parts.
USE_MIXUP = True

# TASER settings (taser)
TASER_NUM_NEIGHBORS = 16
TASER_DIM_TIME = 16
TASER_ATT_HEAD = 4


# =========================
# Yelp path SETTINGS
# =========================
#YELP_PT_PATH = "data/yelp/yelp_hetero_monthly_C60.pt"  # 60 months data
YELP_PT_PATH = "data/yelp/yelp_hetero_monthly_C24.pt"  # 24 months data

# Example: HORIZON=1 means: predict label at (t+1 month) from features up to month t.
HORIZON = 1

# DGNN temporal window length in months
LAGS = 1

# DGNN embedding size
D_EMB = 32
DCRNN_K = 2

# Business k-hop subgraph radius (on geo graph)
K_HOP = 1

# Train/Eval
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# attention / aux losses
INIT_K_FRAC = 0.05
K_MIN = 0.01
K_MAX = 0.2
K_ABS_MIN = 10
K_ABS_MAX = 100
TAU = 0.8
BETA_DIV = 0.1 # 0.1
BETA_METRIC = 0.5 # 0.5
BETA_MIXUP = 1.0 # 1.0

# batching
ANCHORS_PER_SNAPSHOT_TRAIN = 256
ANCHORS_PER_SNAPSHOT_EVAL = 512
MAX_EVAL_ANCHORS = 1024

# Use only first TOTAL_RATIO of months for quick debugging (set 1.0 for all)
TOTAL_RATIO = 1.0 # 1.0

# =========================
# Warm up on the first WARMUP_SNAPSHOTS snapshots (with EPOCHS_WARMUP passes),
# then for each time t, do:
#   - one online update using snapshot t
#   - evaluate on future snapshots t+h for h in TEST_HORIZONS
# Metrics are averaged over the whole rolling period.
WARMUP_SNAPSHOTS = 12          # must be >= 1; typical: max(1, LAGS)
EPOCHS_WARMUP = 10             # number of passes over warmup snapshots
TEST_HORIZONS = (1,) # (1,)     # evaluate t+x if available


# =========================
# plotting config
# =========================
SAVE_CURVE_PLOTS = True        # whether to plot rolling curves
CURVE_OUT_DIR = "p2_curves"    # output folder
CURVE_SAVE_LOG = True          # save raw rolling metrics as .pt

# If you want a sliding window (instead of all history), set TRAIN_WINDOW > 0.
# Example: TRAIN_WINDOW=6 uses only the last 6 snapshots for each online update.
TRAIN_WINDOW = -1

# =========================
# Task choice
# =========================
# TASK1:
TASK = "TASK1"  # "TASK1"

THRESH_TASK1 = 2.1308
# build additional business features by aggregating user->business review edges.
# This injects hetero information into a business-only DGNN backbone.
BIZ_CHANNELS = [0, 1, 2, 3]  # count_log, avg_stars_month, uniq_users_log, total_reviews_log
ELITE_CHANNEL = 3            # user feature channel 3 is elite flag (0/1)

USE_EDGE_RATING = True       # requires store.edge_rating in preprocessing
ADD_RATING_STATS = True      # mean/var rating per business
ADD_ELITE_FRAC = True        # fraction of elite reviewers per business
ADD_LOG_COUNT = True         # log review count per business

# =========================
# FAISS options
# =========================
USE_FAISS_GPU = True
FAISS_TOPM = K_ABS_MAX * 10
FAISS_UPDATE_EVERY = 1
FAISS_UNION_KHOP = True
FAISS_MAX_CAND = 2048
FAISS_METRIC = "ip"
FAISS_REQUIRE_TORCH_GPU = True


# =========================
# Avoid expensive unique/sort on GPU for every anchor.
# =========================
def _union_faiss_and_khop(
    cand_ids: torch.Tensor,
    subset: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    if cand_ids.numel() == 0:
        cand_ids = subset
    if cand_ids.numel() == 0:
        return cand_ids

    subset_u = subset
    if node_mask is not None and subset_u.numel() > 0:
        subset_u = subset_u[node_mask[subset_u]]
    if subset_u.numel() == 0:
        return cand_ids

    # bool seen mask: O(N) and cheap compared to torch.unique/sort.
    seen = torch.zeros((num_nodes,), device=cand_ids.device, dtype=torch.bool)
    seen[cand_ids] = True
    add = subset_u[~seen[subset_u]]
    if add.numel() > 0:
        cand_ids = torch.cat([cand_ids, add], dim=0)
    return cand_ids


def x_to_batched_sequence(x: torch.Tensor, lags: int) -> torch.Tensor:
    # Supports flattened [N, lags*F]
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
    # Precompute all k-hop subsets on CPU once
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


# ============================================================
# Yelp DGNN adapter: hetero aggregation into business features
# ============================================================
@torch.no_grad()
def build_hetero_adapter_features(
    data_t: HeteroData,
    elite_channel: int = 3,
    use_edge_rating: bool = True,
    add_rating_stats: bool = True,
    add_elite_frac: bool = True,
    add_log_count: bool = True,
) -> torch.Tensor:
    """
    Build per-business extra features using user nodes + review edges.
    Output: [Nb, Fu + extras].
    """
    Nb = data_t["business"].x.size(0)
    Fu = data_t["user"].x.size(1)

    store = data_t["user", "rev", "business"]
    e = store.edge_index  # [2, E] (user->business)

    if e.numel() == 0:
        f_extra = Fu + (2 if add_rating_stats else 0) + (1 if add_elite_frac else 0) + (1 if add_log_count else 0)
        return torch.zeros((Nb, f_extra), dtype=torch.float32)

    u = e[0]
    b = e[1]

    x_u = data_t["user"].x.float()
    msg_u = x_u[u]  # [E, Fu]

    sum_u = torch.zeros((Nb, Fu), dtype=torch.float32)
    sum_u.index_add_(0, b, msg_u)

    cnt = torch.zeros((Nb,), dtype=torch.float32)
    cnt.index_add_(0, b, torch.ones((b.size(0),), dtype=torch.float32))

    mean_u = sum_u / cnt.clamp_min(1.0).unsqueeze(-1)
    extras = [mean_u]

    if add_rating_stats:
        if use_edge_rating and hasattr(store, "edge_rating") and store.edge_rating is not None and store.edge_rating.numel() == e.size(1):
            r = store.edge_rating.float()
            sum_r = torch.zeros((Nb,), dtype=torch.float32)
            sum_r2 = torch.zeros((Nb,), dtype=torch.float32)
            sum_r.index_add_(0, b, r)
            sum_r2.index_add_(0, b, r * r)

            mean_r = sum_r / cnt.clamp_min(1.0)
            var_r = (sum_r2 / cnt.clamp_min(1.0)) - mean_r * mean_r
            var_r = var_r.clamp_min(0.0)
        else:
            mean_r = torch.zeros((Nb,), dtype=torch.float32)
            var_r = torch.zeros((Nb,), dtype=torch.float32)

        extras.append(mean_r.unsqueeze(-1))
        extras.append(var_r.unsqueeze(-1))

    if add_elite_frac:
        elite = data_t["user"].x[:, elite_channel].float()
        elite_u = elite[u]
        sum_elite = torch.zeros((Nb,), dtype=torch.float32)
        sum_elite.index_add_(0, b, elite_u)
        frac_elite = sum_elite / cnt.clamp_min(1.0)
        extras.append(frac_elite.unsqueeze(-1))

    if add_log_count:
        extras.append(torch.log1p(cnt).unsqueeze(-1))

    return torch.cat(extras, dim=1).contiguous()


class YelpSnapshot:
    """
      - x: [Nb, lags*F_in]
      - month_idx: index into the original HeteroData list
    """
    def __init__(self, x: torch.Tensor, month_idx: int):
        self.x = x
        self.month_idx = int(month_idx)


@torch.no_grad()
def load_yelp_as_dgnn_snapshots(
    pt_path: str,
    lags: int,
    device: torch.device,
) -> Tuple[List[YelpSnapshot], List[HeteroData], Dict, torch.Tensor, torch.Tensor, int]:
    """
    Load Yelp .pt and convert to DGNN snapshots.
    """
    obj = torch.load(pt_path, map_location="cpu")
    data_list: List[HeteroData] = obj["data_list"]
    meta = obj.get("meta", {})

    # optionally shorten for debug
    if TOTAL_RATIO < 1.0:
        T0 = len(data_list)
        data_list = data_list[: max(2, int(TOTAL_RATIO * T0))]

    edge_index = data_list[0]["business", "geo", "business"].edge_index.to(device)
    edge_weight = data_list[0]["business", "geo", "business"].edge_weight.to(device)

    xb_all = []
    for d in data_list:
        xb = d["business"].x.float()
        if BIZ_CHANNELS is not None:
            xb = xb[:, BIZ_CHANNELS]

        extra = build_hetero_adapter_features(
            d,
            elite_channel=ELITE_CHANNEL,
            use_edge_rating=USE_EDGE_RATING,
            add_rating_stats=ADD_RATING_STATS,
            add_elite_frac=ADD_ELITE_FRAC,
            add_log_count=ADD_LOG_COUNT,
        )
        xb_all.append(torch.cat([xb, extra], dim=1))

    snapshots: List[YelpSnapshot] = []
    base_t_offset = lags - 1
    for month_idx in range(base_t_offset, len(data_list)):
        x_lag = torch.stack([xb_all[month_idx - i] for i in reversed(range(lags))], dim=1)
        x_flat = x_lag.reshape(x_lag.size(0), -1).contiguous()
        snapshots.append(YelpSnapshot(x_flat, month_idx))

    return snapshots, data_list, meta, edge_index, edge_weight, base_t_offset

@torch.no_grad()
def label_task2_activity_persistence(
    data_future: HeteroData,
    biz_subset: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Label=1 if avg review count in region > threshold.
    """
    #biz_subset = biz_subset.to(device=device)
    biz_subset_cpu = biz_subset.cpu()

    # business feature: count_log = log1p(#reviews in month t)
    # you stored it in business.x[:, 0]
    x = data_future["business"].x[biz_subset_cpu, 0]  # log1p(count)
    avg_log_count = x.mean()

    return (avg_log_count > threshold).long()

# ============================================================
# DGNN backbone
# ============================================================
class DGNNEncoder(nn.Module):
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
            raise ValueError(f"Unknown MODEL_NAME={MODEL_NAME!r}.")

    def forward(self, X, edge_index, edge_weight=None):
        if hasattr(self.cell, "reset_state"):
            self.cell.reset_state()

        H = None
        for t in range(X.size(1)):
            x_t = X[0, t]
            try:
                H = self.cell(x_t, edge_index, edge_weight, H)
            except TypeError:
                H = self.cell(x_t, edge_index, edge_weight)
        return H


class SubgraphHead(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, sub_rep: torch.Tensor) -> torch.Tensor:
        return self.lin(sub_rep)


class BinaryCEHead(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.lin(rep)


def pool_subgraph_mean(emb_nodes: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    return emb_nodes[subset].mean(dim=0)


def ce_margin_score(logits2: torch.Tensor) -> torch.Tensor:
    return logits2[..., 1] - logits2[..., 0]


# ============================================================
# eval_epoch: patched labels + snapshot month mapping
# ============================================================
@torch.inference_mode()
def eval_epoch(
    snapshots: List[YelpSnapshot],
    data_list: List[HeteroData],
    encoder: DGNNEncoder,
    head: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    khop_subsets: List[torch.Tensor],
    horizon: int,
    tau: float,
    lags: int,
    device: torch.device,
    t_start: int,
    t_end: int,
    anchors_per_snapshot: int,
    max_eval_anchors: int,
    k_eval: int,
    f1_thr: float,
    attention: Optional[QKOnlySoftTopKAttention] = None,
    query_embed: Optional[nn.Embedding] = None,
    retriever: Optional[FaissGpuRetriever] = None,
    use_faiss_gpu: Optional[bool] = None,
) -> Tuple[float, dict]:

    encoder.eval()
    head.eval()
    if USE_ATTN:
        assert attention is not None and query_embed is not None
        attention.eval()
        query_embed.eval()

    _use_faiss = USE_FAISS_GPU if use_faiss_gpu is None else bool(use_faiss_gpu)
    if USE_ATTN and _use_faiss:
        assert retriever is not None

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    steps = 0

    all_scores_cpu = []
    all_labels_cpu = []
    p_at_k_list = []

    t_end_eff = min(t_end, len(snapshots))
    for s in range(t_start, t_end_eff):
        month_idx = snapshots[s].month_idx
        if month_idx + horizon >= len(data_list):
            break

        x = _to_tensor(snapshots[s].x, device=device, dtype=torch.float)
        X = x_to_batched_sequence(x, lags=lags)
        emb_nodes = encoder(X, edge_index, edge_weight)
        N = emb_nodes.size(0)

        node_mask = None
        candidates = torch.arange(N, device=device)

        cand = candidates
        if max_eval_anchors is not None and max_eval_anchors > 0 and cand.numel() > int(max_eval_anchors):
            perm = torch.randperm(cand.numel(), device=device)
            cand = cand[perm[: int(max_eval_anchors)]]
        if anchors_per_snapshot >= 0 and anchors_per_snapshot < cand.numel():
            perm = torch.randperm(cand.numel(), device=device)
            anchor_indices = cand[perm[: anchors_per_snapshot]]
        else:
            anchor_indices = cand

        snapshot_scores = torch.full((N,), float("-inf"), device=device)
        snapshot_labels = torch.zeros((N,), device=device)

        loss_t = 0.0

        if USE_ATTN and _use_faiss:
            if (s % int(FAISS_UPDATE_EVERY) == 0):
                faiss_valid_idx = torch.arange(N, device=device, dtype=torch.long)
                K_valid = attention.W_k(emb_nodes[faiss_valid_idx]).detach()
                retriever.build(K_valid, valid_idx=faiss_valid_idx, normalize=bool(attention.normalize_qk),
                                require_torch_gpu=FAISS_REQUIRE_TORCH_GPU)

        cand_batch = None
        h_idx = 0 if horizon == 1 else 1
        if USE_ATTN and _use_faiss:
            emb_q = query_embed(torch.tensor(h_idx, device=device))
            emb_a_batch = emb_nodes[anchor_indices]
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

        data_future = data_list[month_idx + horizon]

        for _i, anchor_idx in enumerate(anchor_indices.tolist()):
            subset = khop_subsets[anchor_idx].to(device=device)
            y_class = label_task2_activity_persistence(data_future, subset, THRESH_TASK1, device)
            #print("[debug] pos_ratio =", y_class.float().mean().item(), "N=", y_class.numel())
            if USE_ATTN:
                emb_a = emb_nodes[anchor_idx]
                emb_q = query_embed(torch.tensor(h_idx, device=device))

                if _use_faiss:
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
                        node_mask=None,
                        tau=tau,
                        return_intermediates=True,
                    )

                rep = emb_a + emb_q + context
                logits2 = head(rep)
            else:
                sub_rep = pool_subgraph_mean(emb_nodes, subset)
                logits2 = head(sub_rep)

            loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1).to(device))

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

# ============================================================
# one-step train / one-step eval
# ============================================================
def _sample_anchors(candidates: torch.Tensor, anchors_per_snapshot: int, max_anchors: int) -> torch.Tensor:
    """
    candidates: [N]
    anchors_per_snapshot: if <0 -> all; else sample that many
    max_anchors: hard cap (useful for eval); if <0 -> no cap
    """
    cand = candidates
    if max_anchors is not None and max_anchors > 0 and cand.numel() > int(max_anchors):
        perm = torch.randperm(cand.numel(), device=cand.device)
        cand = cand[perm[: int(max_anchors)]]
    if anchors_per_snapshot is not None and anchors_per_snapshot >= 0 and anchors_per_snapshot < cand.numel():
        perm = torch.randperm(cand.numel(), device=cand.device)
        return cand[perm[: int(anchors_per_snapshot)]]
    return cand


def _get_h_idx(h: int) -> int:
    # uses 2 horizon embeddings: 1->0, else ->1
    return 0 if int(h) == 1 else 1


@torch.inference_mode()
def eval_one_snapshot(
    *,
    snap: YelpSnapshot,
    data_list: List[HeteroData],
    encoder: DGNNEncoder,
    head: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    khop_subsets: List[torch.Tensor],
    horizon: int,
    tau: float,
    lags: int,
    device: torch.device,
    anchors_per_snapshot: int,
    max_eval_anchors: int,
    k_eval: int,
    f1_thr: float,
    attention: Optional[QKOnlySoftTopKAttention] = None,
    query_embed: Optional[nn.Embedding] = None,
    retriever: Optional[FaissGpuRetriever] = None,
    use_faiss_gpu: Optional[bool] = None,
) -> Tuple[float, dict]:
    """
    Evaluate on exactly ONE snapshot (time t), predicting label at (t+horizon).
    Returns: (avg_loss, metrics_dict)
    """
    encoder.eval()
    head.eval()
    if USE_ATTN:
        assert attention is not None and query_embed is not None
        attention.eval()
        query_embed.eval()

    _use_faiss = USE_FAISS_GPU if use_faiss_gpu is None else bool(use_faiss_gpu)
    if USE_ATTN and _use_faiss:
        assert retriever is not None

    month_idx = snap.month_idx
    if month_idx + horizon >= len(data_list):
        return 0.0, {"auc": 0.5, "f1": 0.0, "p@k": 0.0}

    ce = nn.CrossEntropyLoss()

    x = _to_tensor(snap.x, device=device, dtype=torch.float)
    X = x_to_batched_sequence(x, lags=lags)
    emb_nodes = encoder(X, edge_index, edge_weight)
    N = emb_nodes.size(0)

    candidates = torch.arange(N, device=device)
    anchor_indices = _sample_anchors(candidates, anchors_per_snapshot, max_eval_anchors)

    snapshot_scores = torch.full((N,), float("-inf"), device=device)
    snapshot_labels = torch.zeros((N,), device=device)

    # (optional) FAISS rebuild on this snapshot
    if USE_ATTN and _use_faiss:
        faiss_valid_idx = torch.arange(N, device=device, dtype=torch.long)
        K_valid = attention.W_k(emb_nodes[faiss_valid_idx]).detach()
        retriever.build(
            K_valid,
            valid_idx=faiss_valid_idx,
            normalize=bool(attention.normalize_qk),
            require_torch_gpu=FAISS_REQUIRE_TORCH_GPU,
        )

    cand_batch = None
    h_idx = _get_h_idx(horizon)
    if USE_ATTN and _use_faiss:
        emb_q = query_embed(torch.tensor(h_idx, device=device))
        emb_a_batch = emb_nodes[anchor_indices]
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

    data_future = data_list[month_idx + horizon]

    loss_sum = 0.0
    scores_list: List[float] = []
    labels_list: List[float] = []

    for _i, anchor_idx in enumerate(anchor_indices.tolist()):
        subset = khop_subsets[anchor_idx].to(device=device)
        y_class = label_task2_activity_persistence(data_future, subset, THRESH_TASK1, device)
        #print("[debug] pos_ratio =", y_class.float().mean().item(), "N=", y_class.numel())

        if USE_ATTN:
            emb_a = emb_nodes[anchor_idx]
            emb_q = query_embed(torch.tensor(h_idx, device=device))

            if _use_faiss:
                cand_ids = cand_batch[_i].view(-1)
                if FAISS_UNION_KHOP:
                    cand_ids = _union_faiss_and_khop(cand_ids, subset, None, N)
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
                    node_mask=None,
                    tau=tau,
                    return_intermediates=True,
                )
            rep = emb_a + emb_q + context
            logits2 = head(rep)
        else:
            sub_rep = pool_subgraph_mean(emb_nodes, subset)
            logits2 = head(sub_rep)

        loss_sum += float(ce(logits2.view(1, 2), y_class.view(1).to(device)).item())

        score = float(ce_margin_score(logits2).detach())
        snapshot_scores[anchor_idx] = torch.tensor(score, device=device)
        snapshot_labels[anchor_idx] = y_class.detach().float()

        scores_list.append(score)
        labels_list.append(float(y_class))

    avg_loss = loss_sum / max(1, anchor_indices.numel())
    if len(scores_list) == 0:
        return avg_loss, {"auc": 0.5, "f1": 0.0, "p@k": 0.0}

    scores_cpu = torch.tensor(scores_list, dtype=torch.float32)
    labels_cpu = torch.tensor(labels_list, dtype=torch.float32)

    metrics = {
        "auc": binary_auc_from_logits(scores_cpu, labels_cpu),
        "f1": binary_f1_from_logits(scores_cpu, labels_cpu, thr=f1_thr),
        "p@k": float(precision_at_k_from_logits(snapshot_scores, snapshot_labels, k=k_eval)),
    }
    return avg_loss, metrics


def _train_step(
    *,
    snap: YelpSnapshot,
    data_list: List[HeteroData],
    encoder: DGNNEncoder,
    head: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    khop_subsets: List[torch.Tensor],
    horizon: int,
    tau: float,
    lags: int,
    device: torch.device,
    anchors_per_snapshot: int,
    attention: Optional[QKOnlySoftTopKAttention],
    query_embed: Optional[nn.Embedding],
    retriever: Optional[FaissGpuRetriever],
    use_faiss_gpu: bool,
    opt: torch.optim.Optimizer,
    metric_loss: Optional[MetricLoss],
    mixup: Optional[MixupWithMemory],
    beta_metric: float,
    beta_mixup: float,
) -> float:
    """
    One online update on snapshot t, using label from (t+horizon).
    Returns scalar loss.
    """
    encoder.train()
    head.train()
    if USE_ATTN:
        assert attention is not None and query_embed is not None
        attention.train()
        query_embed.train()
        if USE_MIXUP and metric_loss is not None:
            metric_loss.train()

    month_idx = snap.month_idx
    if month_idx + horizon >= len(data_list):
        return 0.0

    ce = nn.CrossEntropyLoss()

    x = _to_tensor(snap.x, device=device, dtype=torch.float)
    X = x_to_batched_sequence(x, lags=lags)
    emb_nodes = encoder(X, edge_index, edge_weight)
    N = emb_nodes.size(0)

    candidates = torch.arange(N, device=device)
    anchor_indices = _sample_anchors(candidates, anchors_per_snapshot, -1)

    # (optional) FAISS rebuild on this snapshot
    if USE_ATTN and use_faiss_gpu:
        faiss_valid_idx = torch.arange(N, device=device, dtype=torch.long)
        K_valid = attention.W_k(emb_nodes[faiss_valid_idx]).detach()
        retriever.build(
            K_valid,
            valid_idx=faiss_valid_idx,
            normalize=bool(attention.normalize_qk),
            require_torch_gpu=FAISS_REQUIRE_TORCH_GPU,
        )

    cand_batch = None
    h_idx = _get_h_idx(horizon)
    if USE_ATTN and use_faiss_gpu:
        emb_q = query_embed(torch.tensor(h_idx, device=device))
        emb_a_batch = emb_nodes[anchor_indices]
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

    data_now = data_list[month_idx]
    data_future = data_list[month_idx + horizon]

    loss_t = 0.0
    all_context, all_query, all_target = [], [], []

    for _i, anchor_idx in enumerate(anchor_indices.tolist()):
        subset = khop_subsets[anchor_idx].to(device=device)
        y_class = label_task2_activity_persistence(data_future, subset, THRESH_TASK1, device)
        #print("[debug] pos_ratio =", y_class.float().mean().item(), "N=", y_class.numel())

        if USE_ATTN:
            emb_a = emb_nodes[anchor_idx]
            emb_q = query_embed(torch.tensor(h_idx, device=device))

            if use_faiss_gpu:
                cand_ids = cand_batch[_i].view(-1)
                if FAISS_UNION_KHOP:
                    cand_ids = _union_faiss_and_khop(cand_ids, subset, None, N)
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
                _div_attn = attn
                _div_nodes = emb_nodes_c
                _div_mask = None
            else:
                context, attn, soft_mask, scores, Q, K, theta, k = attention(
                    emb_q=emb_q,
                    emb_a=emb_a,
                    emb_nodes=emb_nodes,
                    node_mask=None,
                    tau=tau,
                    return_intermediates=True,
                )
                _div_attn = attn
                _div_nodes = emb_nodes
                _div_mask = None

            rep = emb_a + emb_q + context
            logits2 = head(rep)
            loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1).to(device))

            k_hard = max(2, int(round(float(k.detach().cpu().item()))))
            loss_t = loss_t + BETA_DIV * diversity_loss(_div_attn, _div_nodes, k=k_hard, node_mask=_div_mask)

            all_context.append(context)
            all_query.append(emb_q)
            all_target.append(y_class)
        else:
            sub_rep = pool_subgraph_mean(emb_nodes, subset)
            logits2 = head(sub_rep)
            loss_t = loss_t + ce(logits2.view(1, 2), y_class.view(1).to(device))

    loss_t = loss_t / max(1, anchor_indices.numel())

    if USE_ATTN and USE_MIXUP and mixup is not None and metric_loss is not None and len(all_context) > 0:
        ctx = torch.stack(all_context).to(device)
        qry = torch.stack(all_query).to(device)
        tgt = torch.stack(all_target).long().to(device)

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


# ============================================================
# Warmup + rolling evaluation
# ============================================================
def train_one_run(use_faiss_gpu: Optional[bool] = None):
    """
    Protocol P2:
      - Warmup on first W snapshots (EPOCHS_WARMUP passes)
      - For t from W to the end:
          - online update using snapshot t
          - evaluate on t+{1,2,3} (if exists)
      - Report averaged metrics for each horizon
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _use_faiss = USE_FAISS_GPU if use_faiss_gpu is None else bool(use_faiss_gpu)

    stats = {
        "use_faiss": _use_faiss,
        "warmup_time_s": 0.0,
        "rolling_time_s": 0.0,
        "rolling_metrics": {},  # horizon -> avg metrics
    }

    snapshots, data_list, meta, edge_index, edge_weight, base_t_offset = load_yelp_as_dgnn_snapshots(
        pt_path=YELP_PT_PATH, lags=LAGS, device=device
    )
    S = len(snapshots)

    x0 = snapshots[0].x
    assert x0.dim() == 2 and x0.shape[1] % LAGS == 0
    num_features = int(x0.shape[1] // LAGS)
    num_nodes = int(x0.shape[0])

    khop_subsets = precompute_khop_subsets(edge_index, num_nodes=num_nodes, k_hop=K_HOP)
    encoder = DGNNEncoder(in_channels=num_features, d_emb=D_EMB, K=DCRNN_K).to(device)

    query_embed = None
    attention = None
    metric_loss = None
    mixup = None
    retriever = None

    if USE_ATTN:
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
            k_abs_min=K_ABS_MIN,
            k_abs_max=K_ABS_MAX,
        ).to(device)
        head = BinaryCEHead(D_EMB).to(device)

        if USE_MIXUP:
            metric_loss = MetricLoss(num_classes=2, d_emb=D_EMB, code_size=D_EMB, device=device).to(device)
            mixup = MixupWithMemory(num_classes=2, d_emb=D_EMB, device=device)

        if _use_faiss:
            retriever = FaissGpuRetriever(device=device, metric=FAISS_METRIC)

        params = list(encoder.parameters()) + list(query_embed.parameters()) + list(attention.parameters()) + list(head.parameters())
        if USE_MIXUP:
            params = params + list(metric_loss.parameters())
    else:
        head = SubgraphHead(D_EMB).to(device)
        params = list(encoder.parameters()) + list(head.parameters())

    opt = torch.optim.Adam(params, lr=LR)

    W = max(1, int(WARMUP_SNAPSHOTS))
    W = min(W, S - 1)  # need at least one step after warmup
    Hs = tuple(int(h) for h in TEST_HORIZONS)
    Hmax = max(Hs)

    mode_str = "DIRECT" if not USE_ATTN else ("ATN+MIX+MET" if USE_MIXUP else "ATN")
    print(f"=== Yelp {mode_str} ({MODEL_NAME}) Protocol=P2 TASK={TASK} k-hop={K_HOP} ===")
    print(f"Snapshots={S}, months={len(data_list)}, base_t_offset={base_t_offset}")
    print(f"Warmup: first W={W} snapshots, epochs_warmup={EPOCHS_WARMUP}")
    print(f"Rolling eval horizons={Hs}   FAISS={'ON' if _use_faiss else 'OFF'}")

    # -------------------------
    # Warmup phase (offline)
    # -------------------------
    _sync_if_cuda(device)
    t0 = _now()

    global_step = 0
    for ep in range(int(EPOCHS_WARMUP)):
        tau_ep = TAU * (0.9 ** ep)
        beta_metric_ep = BETA_METRIC * (0.9 ** ep)
        beta_mixup_ep = BETA_MIXUP * (0.9 ** ep)

        losses = []
        for s in range(0, W):
            loss_val = _train_step(
                snap=snapshots[s],
                data_list=data_list,
                encoder=encoder,
                head=head,
                edge_index=edge_index,
                edge_weight=edge_weight,
                khop_subsets=khop_subsets,
                horizon=HORIZON,
                tau=tau_ep,
                lags=LAGS,
                device=device,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_TRAIN,
                attention=attention,
                query_embed=query_embed,
                retriever=retriever,
                use_faiss_gpu=False, #use_faiss_gpu=_use_faiss,
                opt=opt,
                metric_loss=metric_loss,
                mixup=mixup,
                beta_metric=beta_metric_ep,
                beta_mixup=beta_mixup_ep,
            )
            losses.append(loss_val)
            global_step += 1
        print(f"warmup_epoch={ep+1} avg_loss={sum(losses)/max(1,len(losses)):.6f} tau={tau_ep:.4f}")

    _sync_if_cuda(device)
    stats["warmup_time_s"] = _now() - t0

    # -------------------------
    # Rolling phase (online)
    # -------------------------
    # accumulate metrics per horizon.
    acc = {h: {"loss": 0.0, "auc": 0.0, "f1": 0.0, "p@k": 0.0, "n": 0} for h in Hs}

    # initialize per-step series containers (one list per horizon)
    stats["rolling_series"] = {h: [] for h in Hs}
    stats["rolling_loss_series"] = []


    _sync_if_cuda(device)
    t1 = _now()

    # for each time t: update on snapshot t, then evaluate on t+h
    for t in range(W, S - Hmax):
        # online update on time t
        # (optional) a sliding training window: do extra steps on past snapshots
        tau_t = TAU * (0.9 ** (t / max(1, S)))
        beta_metric_t = BETA_METRIC * (0.9 ** (t / max(1, S)))
        beta_mixup_t = BETA_MIXUP * (0.9 ** (t / max(1, S)))

        if TRAIN_WINDOW and int(TRAIN_WINDOW) > 0:
            # track average online training loss across the window
            loss_sum_win = 0.0
            loss_cnt_win = 0

            start = max(0, t - int(TRAIN_WINDOW) + 1)
            for s in range(start, t + 1):
                loss_val = _train_step(
                    snap=snapshots[s],
                    data_list=data_list,
                    encoder=encoder,
                    head=head,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    khop_subsets=khop_subsets,
                    horizon=HORIZON,
                    tau=tau_t,
                    lags=LAGS,
                    device=device,
                    anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_TRAIN,
                    attention=attention,
                    query_embed=query_embed,
                    retriever=retriever,
                    use_faiss_gpu=_use_faiss,
                    opt=opt,
                    metric_loss=metric_loss,
                    mixup=mixup,
                    beta_metric=beta_metric_t,
                    beta_mixup=beta_mixup_t,
                )
                loss_sum_win += float(loss_val)
                loss_cnt_win += 1

            # averaged online loss for this time t
            loss_online = loss_sum_win / max(1, loss_cnt_win)

            # record loss at time t
            stats["rolling_loss_series"].append({"t": int(t), "loss": float(loss_online)})
        else:
            # capture the online training loss at time t
            loss_online = _train_step(
                snap=snapshots[t],
                data_list=data_list,
                encoder=encoder,
                head=head,
                edge_index=edge_index,
                edge_weight=edge_weight,
                khop_subsets=khop_subsets,
                horizon=HORIZON,
                tau=tau_t,
                lags=LAGS,
                device=device,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_TRAIN,
                attention=attention,
                query_embed=query_embed,
                retriever=retriever,
                use_faiss_gpu=_use_faiss,
                opt=opt,
                metric_loss=metric_loss,
                mixup=mixup,
                beta_metric=beta_metric_t,
                beta_mixup=beta_mixup_t,
            )
            # record loss at time t
            stats["rolling_loss_series"].append({"t": int(t), "loss": float(loss_online)})

        # evaluate on future horizons from time t (using snapshot t as input)
        for h in Hs:
            loss_h, m_h = eval_one_snapshot(
                snap=snapshots[t],
                data_list=data_list,
                encoder=encoder,
                head=head,
                edge_index=edge_index,
                edge_weight=edge_weight,
                khop_subsets=khop_subsets,
                horizon=h,
                tau=tau_t,
                lags=LAGS,
                device=device,
                anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_EVAL,
                max_eval_anchors=MAX_EVAL_ANCHORS,
                k_eval=K_EVAL,
                f1_thr=F1_THR,
                attention=attention,
                query_embed=query_embed,
                retriever=retriever,
                use_faiss_gpu=_use_faiss,
            )
            acc[h]["loss"] += float(loss_h)
            acc[h]["auc"] += float(m_h["auc"])
            acc[h]["f1"] += float(m_h["f1"])
            acc[h]["p@k"] += float(m_h["p@k"])
            acc[h]["n"] += 1

            # record per-step metrics for plotting
            stats["rolling_series"][h].append({
                "t": int(t),
                "auc": float(m_h["auc"]),
                "f1": float(m_h["f1"]),
                "p@k": float(m_h["p@k"]),
                "loss": float(loss_online),
            })

        if (t - W + 1) % 1 == 0:
            # light progress print each step (few snapshots). Increase modulus for longer runs.
            msg = [f"t={t}"]
            for h in Hs:
                n = acc[h]["n"]
                if n:
                    msg.append(f"h{h}:AUC={acc[h]['auc']/n:.3f},F1={acc[h]['f1']/n:.3f},P@{K_EVAL}={acc[h]['p@k']/n:.3f}")
            print("  " + " | ".join(msg))

    _sync_if_cuda(device)
    stats["rolling_time_s"] = _now() - t1

    # finalize averages
    for h in Hs:
        n = max(1, acc[h]["n"])
        stats["rolling_metrics"][h] = {
            "avg_loss": acc[h]["loss"] / n,
            "auc": acc[h]["auc"] / n,
            "f1": acc[h]["f1"] / n,
            "p@k": acc[h]["p@k"] / n,
            "n_steps": acc[h]["n"],
        }

    return stats



# ============================================================
# plot rolling metric curves
# ============================================================
def _ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)

def plot_p2_curves(stats: dict, out_dir: str, k_eval: int):
    """
    stats["rolling_series"] expected:
      {horizon: [{"t": int, "auc": float, "f1": float, "p@k": float, "loss": float}, ...], ...}
    Creates 3 PNGs: auc.png, f1.png, p_at_k.png (lines = horizons).
    """
    if "rolling_series" not in stats:
        print("[plot] No rolling_series found; skip plotting.")
        return

    _ensure_dir(out_dir)

    series = stats["rolling_series"]
    horizons = sorted(series.keys())

    def _plot(metric_key: str, title: str, filename: str):
        plt.figure()
        for h in horizons:
            xs = [d["t"] for d in series[h]]
            ys = [d[metric_key] for d in series[h]]
            plt.plot(xs, ys, label=f"h={h}")
        plt.xlabel("rolling time t (snapshot index)")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename}", dpi=200)
        plt.close()

    _plot("auc", "Rolling AUC over time", "auc.png")
    _plot("f1", "Rolling F1 over time", "f1.png")
    _plot("p@k", f"Rolling Precision@{k_eval} over time", "p_at_k.png")

    has_loss = any(("loss" in d) for h in horizons for d in series[h])
    if has_loss:
        _plot("loss", "Rolling loss over time", "loss.png")


if __name__ == "__main__":
    print("\n===== Protocol P2: FAISS OFF =====")
    stats_no = train_one_run(use_faiss_gpu=False)
    if USE_ATTN == True:
        print("\n===== Protocol P2: FAISS ON =====")
        stats_yes = train_one_run(use_faiss_gpu=True)

    print("\n===== P2 Summary =====")

    # save and plot curves
    import os
    if SAVE_CURVE_PLOTS:
        out_dir = CURVE_OUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        if USE_ATTN == True:
            if CURVE_SAVE_LOG:
                torch.save({"no_faiss": stats_no, "faiss": stats_yes}, f"{out_dir}/p2_series.pt")
            plot_p2_curves(stats_no, f"{out_dir}/no_faiss", k_eval=K_EVAL)
            plot_p2_curves(stats_yes, f"{out_dir}/faiss", k_eval=K_EVAL)
            print(f"[P2 curves] saved under: {out_dir}/")
        else:
            if CURVE_SAVE_LOG:
                torch.save({"no_faiss": stats_no}, f"{out_dir}/p2_series.pt")
            plot_p2_curves(stats_no, f"{out_dir}/no_faiss", k_eval=K_EVAL)
            print(f"[P2 curves] saved under: {out_dir}/")

    if USE_ATTN == True:
        for name, st in [("No-FAISS", stats_no), ("FAISS", stats_yes)]:
            print(f"\n{name}: warmup_time={st['warmup_time_s']:.2f}s rolling_time={st['rolling_time_s']:.2f}s")
            for h, m in st["rolling_metrics"].items():
                print(
                    f"  horizon={h} avg_loss={m['avg_loss']:.6f} AUC={m['auc']:.3f} F1={m['f1']:.3f} P@{K_EVAL}={m['p@k']:.3f} steps={m['n_steps']}")
    else:
        for name, st in [("No-FAISS", stats_no),]:
            print(f"\n{name}: warmup_time={st['warmup_time_s']:.2f}s rolling_time={st['rolling_time_s']:.2f}s")
            for h, m in st["rolling_metrics"].items():
                print(
                    f"  horizon={h} avg_loss={m['avg_loss']:.6f} AUC={m['auc']:.3f} F1={m['f1']:.3f} P@{K_EVAL}={m['p@k']:.3f} steps={m['n_steps']}")

