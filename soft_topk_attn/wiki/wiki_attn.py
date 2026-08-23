import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from soft_topk_attn.data.tgb_dataloader import TGBDatasetLoader

from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.diversity_loss_f import diversity_loss
from soft_topk_attn.models.mixup import MixupWithMemory
from soft_topk_attn.models.metric_loss import MetricLoss
from soft_topk_attn.models.metrics_bin import (
    binary_auc_from_logits,
    binary_f1_from_logits,
    precision_at_k_from_logits,
    compute_mrr
)

def _to_tensor(x, device, dtype=torch.float):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

class EdgeWeightProjection(nn.Module):
    def __init__(self, edge_feat_dim):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.lin = nn.Linear(self.edge_feat_dim, 1)

    def forward(self, edge_feat):
        return self.lin(edge_feat)

class DCRNNEncoder(nn.Module):
    """
      - takes a snapshot (PyG Data object)
      - returns node embeddings [N, d_emb]
    """
    def __init__(self, in_channels: int, d_emb: int, d_edge_attr: int, K: int = 2):
        super().__init__()
        self.d_emb = d_emb
        self.d_edge_attr = d_edge_attr
        
        if self.d_edge_attr > 0:
            self.edge_weight_proj = EdgeWeightProjection(edge_feat_dim=d_edge_attr)
        else:
            self.edge_weight_proj = None
        
        self.cell = DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)

    def forward(
        self,
        snapshot: Data,
        edge_label_index
    ) -> torch.Tensor:
        X = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr

        if self.d_edge_attr == 1:
            edge_weight = edge_attr
        elif self.edge_weight_proj is not None:
            edge_weight = self.edge_weight_proj(edge_attr).squeeze(1)
        else:
            edge_weight = None

        H = self.cell(X, edge_index, edge_weight=edge_weight) # [N, d_emb]

        return H

class ClassifierHead(nn.Module):
    """
    Edge-level classifier head to identify the existence of an edge
    Input is a edge representation (two node embeddings, pooled), output 2 logits.
    """
    def __init__(self, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_emb, 2)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        # rep: [d_emb] or [B, d_emb]
        return self.lin(rep)

def run_on_wiki_sequence(
    snapshots,                    # snapshot iterator from WIKI dataset
    encoder: DCRNNEncoder,         # DCRNN encoder: X -> emb_nodes
    transform: RandomLinkSplit,
    attention: QKOnlySoftTopKAttention,
    query_embed: nn.Embedding,     # 2 embeddings: future {1,3}
    head: ClassifierHead,
    metric_loss: MetricLoss,
    mixup: MixupWithMemory,
    optimizer: Optional[torch.optim.Optimizer],
    beta_1: float = 0.1,
    beta_2: float = 0.1,
    tau: float = 0.05,
    lags: int = 12,
    anchors_per_snapshot: int = 32,
    device: Optional[torch.device] = None,
    train: bool = True,
    t_start: int = 0,
    t_end: Optional[int] = None,
    k_eval: int = 50,
    f1_thr: float = 0.5,
    compute_metrics: bool = False,
    eval_all_nodes_for_p_at_k: bool = False,
    max_eval_anchors: int = 256,
):
    """
    Train/eval over a sequence of Wiki snapshots (no batching).

    Metrics (if compute_metrics=True):
      - AUC, F1: micro over all evaluated anchors across all t.
      - Precision@k: per-snapshot per-future over node scores.
        For fair P@k vs baseline, set eval_all_nodes_for_p_at_k=True and anchors_per_snapshot=-1 in eval.
    """
    if device is None:
        device = torch.device("cpu")
    if t_end is None:
        t_end = len(snapshots)

    encoder.to(device)
    attention.to(device)
    query_embed.to(device)
    head.to(device)
    metric_loss.to(device)

    if train:
        encoder.train()
        attention.train()
        query_embed.train()
        head.train()
        metric_loss.train()
        mixup.train()
    else:
        encoder.eval()
        attention.eval()
        query_embed.eval()
        head.eval()
        metric_loss.eval()
        mixup.eval()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    all_logits = []
    all_label = []
    all_p_at_k = []
    all_mrr = []

    start_time = time.perf_counter()

    for t in range(t_start, t_end):
        snapshot = next(snapshots)

        snapshot, _, _ = transform(snapshot)

        snapshot.to(device)

        edge_label_index = snapshot.edge_label_index
        edge_label = snapshot.edge_label

        emb_nodes = encoder(snapshot, edge_label_index)
        # number of pairs (both positive and negative)
        N = edge_label_index.size(1)

        node_mask = getattr(snapshot, "node_mask", None)

        if node_mask is not None:
            node_mask = _to_tensor(node_mask, device=device, dtype=torch.bool).view(-1)

        if node_mask is None:
            candidates = torch.arange(N, device=device)
        else:
            candidates = torch.nonzero(node_mask, as_tuple=False).view(-1)
        
        if candidates.numel() == 0:
            continue

        # Choose anchors. In eval, cap anchors to bound memory/compute.
        if (not train) and compute_metrics:
            candidates_eval = candidates
            if max_eval_anchors is not None and max_eval_anchors > 0 and candidates_eval.numel() > int(max_eval_anchors):
                perm = torch.randperm(candidates_eval.numel(), device=device)
                candidates_eval = candidates_eval[perm[:int(max_eval_anchors)]]
            anchor_indices = candidates_eval
            if anchors_per_snapshot >= 0 and anchors_per_snapshot < anchor_indices.numel():
                perm = torch.randperm(anchor_indices.numel(), device=device)
                anchor_indices = anchor_indices[perm[:anchors_per_snapshot]]
        else:
            if anchors_per_snapshot < 0 or anchors_per_snapshot >= candidates.numel():
                anchor_indices = candidates
            else:
                perm = torch.randperm(candidates.numel(), device=device)
                anchor_indices = candidates[perm[:anchors_per_snapshot]]

        loss_t = 0.0


        logits = []
        labels = []
        # same shape as edge indicies: [2, N]
        edge_indicies = [[], []]
        
        all_anchor_labels_list = []
        all_anchor_context_list = []
        all_anchor_query_list = []

        for anchor_idx in anchor_indices.tolist():
            emb_node_head = emb_nodes[edge_label_index[0, anchor_idx]]  # [d_emb]
            emb_node_tail = emb_nodes[edge_label_index[1, anchor_idx]] # [d_emb]

            # dummy query
            q = 0

            emb_q = query_embed(torch.tensor(q, device=device))  # [d_emb]

            # get evidence for head
            out = attention(
                emb_q=emb_q,
                emb_a=emb_node_head,
                emb_nodes=emb_nodes,
                node_mask=node_mask,
                tau=tau,
                return_intermediates=True,
            )
            
            context_head, attn_head, soft_mask_head, scores_head, Q_head, K_head, theta_head, k = out

            # get evidence for tail
            out = attention(
                emb_q=emb_q,
                emb_a=emb_node_tail,
                emb_nodes=emb_nodes,
                node_mask=node_mask,
                tau=tau,
                return_intermediates=True,
            )

            context_tail, attn_tail, soft_mask_tail, scores_tail, Q_tail, K_tail, theta_tail, k = out

            # pool embeddings evidence, and attention
            emb_a = torch.mean(torch.stack([emb_node_head, emb_node_tail], dim=0), dim=0)
            context = torch.mean(torch.stack([context_head, context_tail], dim=0), dim=0)

            rep = emb_a + emb_q + context  # [d_emb]

            logit = head(rep)

            label = edge_label[anchor_idx].unsqueeze(0).to(torch.int64)

            all_anchor_labels_list.append(label)
            all_anchor_context_list.append(context)
            all_anchor_query_list.append(emb_q)

            k_hard = int(round(float(k.detach().cpu().item())))
            k_hard = max(2, k_hard)
            
            div_head = diversity_loss(attn_head, emb_nodes, k=k_hard, node_mask=node_mask)
            div_tail = diversity_loss(attn_tail, emb_nodes, k=k_hard, node_mask=node_mask)
            loss_t = loss_t + beta_1 * (div_head + div_tail)

            base = ce(logit.view(1, 2), label.view(1))
            loss_t = loss_t + base

            if compute_metrics:
                logits.append(logit.detach())
                labels.append(label.detach())
                all_logits.append(logit.detach())
                all_label.append(label.detach())
                edge_indicies[0].append(edge_label_index[0, anchor_idx].detach())
                edge_indicies[1].append(edge_label_index[1, anchor_idx].detach())
            
        if compute_metrics:
            edge_indicies = torch.tensor(edge_indicies).to(device)
            logits = torch.stack(logits, dim=0)
            preds = torch.argmax(logits, dim=1)
            labels = torch.concat(labels)
            all_mrr.append(compute_mrr(edge_indicies, labels, preds, 1, snapshot.num_nodes))

            if torch.isfinite(logits).any() and eval_all_nodes_for_p_at_k:
                all_p_at_k.append(precision_at_k_from_logits(preds, labels, k=k_eval))

        # if train: 
            all_anchor_context_list = torch.stack(all_anchor_context_list).to(device)
            all_anchor_query_list = torch.stack(all_anchor_query_list).to(device)
            all_anchor_labels_list = torch.cat(all_anchor_labels_list).to(device)

            out = mixup.get_mixup_samples(all_anchor_context_list, all_anchor_query_list, all_anchor_labels_list)
            mixup_context, mixup_target, target_i, target_j, mixup_query = out

            metric_loss_t = metric_loss(all_anchor_context_list,
                                    mixup_context,
                                    all_anchor_query_list,
                                    mixup_query,
                                    all_anchor_labels_list,
                                    target_i,
                                    target_j)
            
            metric_loss_t = beta_2 * torch.sum(metric_loss_t)
        
            # perform prediction with head on mixup examples
            mixup_rep = mixup_context + mixup_query
            logit = head(mixup_rep)

            mixup_loss_t = ce(logit, mixup_target)  
            mixup_loss_t = torch.sum(mixup_loss_t)
            
            loss_t = loss_t + metric_loss_t + mixup_loss_t
     
        loss_t = loss_t / max(1, anchor_indices.numel())

        if train:
            if optimizer is None:
                raise ValueError("optimizer must be provided when train=True")
            optimizer.zero_grad(set_to_none=True)
            loss_t.backward()
            optimizer.step()

        total_loss += float(loss_t.detach().cpu().item())
        count += 1

    end_time = time.perf_counter()
    run_time = end_time - start_time

    if train:
        print(f"training time: {run_time:.4f}")

    avg_loss = total_loss / max(count, 1)

    if not compute_metrics:
        return avg_loss

    if len(all_logits) == 0:
        metrics = {
            "auc": 0.5,
            "f1":  0.0,
            "p@k": 0.0,
            "mrr": 0.0,
        }
        return avg_loss, metrics
    
    all_logits = torch.stack(all_logits, dim=0)
    all_pred = torch.argmax(all_logits, dim=1)
    all_label = torch.concat(all_label)

    metrics = {
        "auc": binary_auc_from_logits(all_pred, all_label),
        "f1":  binary_f1_from_logits(all_pred, all_label, thr=f1_thr),
        "p@k": float(sum(all_p_at_k) / max(1, len(all_p_at_k))),
        "mrr": float(sum(all_mrr) / max(1, len(all_mrr)))
    }

    return avg_loss, metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TGBDatasetLoader(name="tgbl-wiki")

    # ---- model components ----
    num_node_feats = dataset.num_node_feats
    num_edge_feats = dataset.num_edge_feats
    d_emb = 32
    code_size = 32
    num_classes = 2

    # negative sampler for link pred task
    transform = RandomLinkSplit(
        num_val = 0,
        num_test = 0,
        add_negative_train_samples = True,
        neg_sampling_ratio = 1.0,
    )

    encoder = DCRNNEncoder(in_channels=num_node_feats, d_emb=d_emb, d_edge_attr=num_edge_feats, K=2)

    query_embed = nn.Embedding(1, d_emb)

    # Your attention module (Newton solver version)
    attention = QKOnlySoftTopKAttention(
        d_in=d_emb,
        d_out=d_emb,
        tau=0.05,
        init_k_frac=0.05,
        normalize_qk=False,
        newton_iters=15,
        newton_damping=1.0,
    )

    metric_loss = MetricLoss(num_classes=num_classes, d_emb=d_emb, code_size=code_size, device=device)

    head = ClassifierHead(d_emb=d_emb)

    mixup = MixupWithMemory(num_classes=num_classes, d_emb=d_emb, device=device)

    params = list(encoder.parameters()) + list(query_embed.parameters()) + list(attention.parameters()) + list(head.parameters()) + list(metric_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    # ---- task settings ----
    beta_1 = 0.1
    beta_2 = 0.1
    tau = 0.05
    warmup_steps = 1000

    # ---- split ----
    T = 500
    train_end = int(0.7 * T)
    test_end = T

# ---- train / eval ----
k_eval = 50  # Precision@k uses k=50 (same as baseline). Change if you want.
for epoch in range(1, 11):
    snapshots = dataset.get_snapshots(warmup_steps=warmup_steps)

    train_loss = run_on_wiki_sequence(
        snapshots=snapshots,
        encoder=encoder,
        transform=transform,
        attention=attention,
        query_embed=query_embed,
        head=head,
        optimizer=optimizer,
        metric_loss=metric_loss,
        mixup=mixup,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau,
        anchors_per_snapshot=32,
        device=device,
        train=True,
        t_start=0,
        t_end=train_end,
        compute_metrics=False,
    )

    with torch.inference_mode(): #reduce GPU memory consumption
        test_loss, test_m = run_on_wiki_sequence(
        snapshots=snapshots,
        encoder=encoder,
        transform=transform,
        attention=attention,
        query_embed=query_embed,
        head=head,
        optimizer=None,
        metric_loss=metric_loss,
        mixup=mixup,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau,
        anchors_per_snapshot=-1,  # score all nodes as anchors for fair P@k
        device=device,
        train=False,
        t_start=train_end,
        t_end=test_end,
        k_eval=k_eval,
        compute_metrics=True,
        eval_all_nodes_for_p_at_k=True,
    )
    
    del snapshots

    print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} "
            f"test(AUC={test_m['auc']:.3f}, F1={test_m['f1']:.3f}, P@50={test_m['p@k']:.3f},  MRR={test_m['mrr']:.3f}) "
        )
