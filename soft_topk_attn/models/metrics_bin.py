import torch
from torch_scatter import scatter_mean


@torch.no_grad()
def binary_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    """
    logits, labels: [M] (flattened)
    threshold on sigmoid(logits)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).to(torch.int64)
    labels = labels.to(torch.int64)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


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


@torch.no_grad()
def precision_at_k_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, k: int
) -> float:
    """
    logits, labels: [N] for ONE snapshot (one horizon)
    Precision@k: among top-k predicted nodes, fraction truly positive.
    """
    N = logits.numel()
    if N == 0:
        return 0.0
    k = int(min(max(k, 1), N))
    topk_idx = torch.topk(logits, k=k, largest=True).indices
    return labels[topk_idx].float().mean().item()


@torch.no_grad()
def compute_mrr(edge_label_index, edge_label, pred, num_neg_per_node, num_nodes):

    src_lst = torch.unique(edge_label_index[0], sorted=True)
    num_src = len(src_lst)
    
    edge_pos = edge_label_index[:, edge_label == 1]
    edge_neg = edge_label_index[:, edge_label == 0]
    
    #prediction scores of all pos and neg edges
    p_pos = pred[edge_label == 1]
    p_neg = pred[edge_label == 0]  
    best_p_pos = scatter_mean(src=p_pos, index=edge_pos[0], dim_size=num_nodes)    
    best_p_pos_by_src = best_p_pos[src_lst]    
    uni, counts = torch.unique(edge_neg[0], sorted=True, return_counts=True)

    # edge_neg (src, dst) are sorted by src
    # find index of first occurence of each src in edge_neg[0]
    first_occ_idx = torch.cumsum(counts, dim=0) - counts
    add = torch.arange(num_neg_per_node, device=first_occ_idx.device)
    score_idx = first_occ_idx.view(-1, 1) + add.view(1, -1)
    p_neg_by_src = p_neg[score_idx] #(num_users, num_neg_per_node)

    if len(p_neg_by_src) < num_src:
        compare = (p_neg_by_src >= best_p_pos_by_src.view(num_src, 1)[:len(p_neg_by_src)]).float()
    else:
        compare = (p_neg_by_src >= best_p_pos_by_src.view(num_src, 1)).float()
    
    #counts 1 + how many negative edge from src has higher score than best_p
    #if there is no such negative edge, rank=1
    rank_by_user = compare.sum(axis=1) + 1 #(num_users, )
    mrr = float(torch.mean(1 / rank_by_user))    
    return mrr