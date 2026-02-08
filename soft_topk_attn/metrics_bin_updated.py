# metrics_bin.py
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def roc_auc_from_logits(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    if len(set(labels.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def ap_from_logits(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    if len(set(labels.tolist())) < 2:
        return float(labels.mean())
    return float(average_precision_score(labels, scores))

@torch.no_grad()
def precision_at_k_from_logits(scores: torch.Tensor, labels: torch.Tensor, k: int = 50) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    n = scores.numel()
    if n == 0:
        return 0.0

    k = min(k, n)
    topk_idx = torch.topk(scores, k).indices
    topk_labels = labels[topk_idx]

    precision = topk_labels.sum().item() / float(k)
    return float(precision)

@torch.no_grad()
def recall_at_k_from_logits(scores: torch.Tensor, labels: torch.Tensor, k: int = 50) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    n = scores.numel()
    if n == 0:
        return 0.0

    k = min(k, n)
    topk_idx = torch.topk(scores, k).indices
    topk_labels = labels[topk_idx]

    total_pos = labels.sum().item()
    if total_pos == 0:
        return 0.0

    recall = topk_labels.sum().item() / total_pos
    return float(recall)

@torch.no_grad()
def f1_from_logits(
    scores: torch.Tensor,      # margin scores: logit1 - logit0
    labels: torch.Tensor       # {0,1}
) -> float:
    # model decision boundary is margin > 0
    preds = (scores > 0.0).long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    if precision + recall < 1e-8:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

@torch.no_grad()
def ndcg_at_k_from_logits(scores: torch.Tensor, labels: torch.Tensor, k: int = 50) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    n = scores.numel()
    if n == 0:
        return 0.0

    k = min(k, n)
    sorted_idx = torch.argsort(scores, descending=True)
    topk_idx = sorted_idx[:k]
    topk_labels = labels[topk_idx]

    gains = (2 ** topk_labels - 1)
    discounts = torch.log2(torch.arange(2, k + 2).float())
    dcg = (gains / discounts).sum().item()

    ideal_labels = torch.sort(labels, descending=True)[0][:k]
    ideal_gains = (2 ** ideal_labels - 1)
    idcg = (ideal_gains / discounts).sum().item()

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

@torch.no_grad()
def binary_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    AUC via rank statistic (equivalent to Mann–Whitney U).
    Works without sklearn. O(M log M) due to sorting.

    logits, labels: [M]
    Returns 0.5 if only one class present.
    """
    labels = labels.to(torch.int64)
    pos = (labels == 1)
    neg = (labels == 0)
    n_pos = pos.sum().item()
    n_neg = neg.sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    scores = logits  # monotonic with probs, ok for AUC
    order = torch.argsort(scores)  # ascending
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float)

    # handle ties by average rank
    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    # find tie groups
    dif = torch.diff(sorted_scores)
    tie_starts = torch.where(dif != 0)[0] + 1
    tie_starts = torch.cat([torch.tensor([0], device=scores.device), tie_starts, torch.tensor([scores.numel()], device=scores.device)])
    # average ranks in each tie segment
    for i in range(tie_starts.numel() - 1):
        a = int(tie_starts[i].item())
        b = int(tie_starts[i + 1].item())
        if b - a > 1:
            avg = sorted_ranks[a:b].mean()
            sorted_ranks[a:b] = avg
    # write back
    ranks[order] = sorted_ranks

    sum_ranks_pos = ranks[pos].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)