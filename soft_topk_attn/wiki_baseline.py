import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.data import Data

from tgb_dataloader import TGBDatasetLoader
from metrics_bin import binary_auc_from_logits, binary_f1_from_logits, precision_at_k_from_logits, compute_mrr

class EdgeWeightProjection(nn.Module):
    def __init__(self, edge_feat_dim):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.lin = nn.Linear(self.edge_feat_dim, 1)

    def forward(self, edge_feat):
        return self.lin(edge_feat)

class LinkPredictor(nn.Module):
    '''
    Classification head for link prediction
    '''
    def forward(self, x, edge_label_index):
        edge_head = x[edge_label_index[0]]
        edge_tail = x[edge_label_index[1]]
        #compute dot product to get a prediction per edge
        return (edge_head * edge_tail).sum(dim=-1)

class DCRNNLinkPredictor(nn.Module):
    """
      - uses DCRNN recurrent cell step-by-step over the lags dimension
      - returns node embeddings from the last time step: [N, d_emb]
    """
    def __init__(self, in_channels: int, d_emb: int, d_edge_attr: int, K: int = 2):
        super().__init__()
        self.d_emb = d_emb
        self.d_edge_attr = d_edge_attr
        self.edge_weight_proj = EdgeWeightProjection(edge_feat_dim=d_edge_attr)
        self.cell = DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)
        self.link_predict = LinkPredictor()

    def forward(
        self,
        snapshot: Data,
        edge_label_index
    ) -> torch.Tensor:
        X = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr

        edge_weight = self.edge_weight_proj(edge_attr).squeeze(1)

        H = self.cell(X, edge_index, edge_weight=edge_weight) # [N, d_emb]

        predict = self.link_predict(H, edge_label_index)

        return predict, H


@torch.inference_mode()
def eval_epoch(
    model: nn.Module,
    snapshots,
    device: torch.device,
    transform,
    t_start: int,
    t_end: int,
    k_eval: int = 50,
    f1_thr: float = 0.5,
):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    count = 0

    all_logits = []
    all_labels = []
    all_p_at_k = []
    all_mrr = []
    
    # For +3 label: use y at t+2 (because y is "next step")
    for t in range(t_start, t_end):
        snapshot = next(snapshots)

        # negative edge sample
        snapshot, _, _ = transform(snapshot)

        snapshot.to(device)

        edge_label_index = snapshot.edge_label_index
        edge_label = snapshot.edge_label

        logits, H = model(snapshot, edge_label_index)

        loss = bce(logits, edge_label)

        total_loss += float(loss.item())
        count += 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(edge_label.detach().cpu())

        all_p_at_k.append(precision_at_k_from_logits(logits, edge_label, k=k_eval))

        mrr = compute_mrr(edge_label_index, edge_label, logits, 1, snapshot.num_nodes)
        
        all_mrr.append(mrr)

    avg_loss = total_loss / max(1, count)

    if count == 0:
        metrics = {
            "auc": 0.5, "f1": 0.0, "p@k": 0.0, "mrr": 0.0
        }
        return avg_loss, metrics

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = {
        "auc": binary_auc_from_logits(logits, labels),
        "f1":  binary_f1_from_logits(logits, labels, thr=f1_thr),
        "p@k": float(sum(all_p_at_k) / max(1, len(all_p_at_k))),
        "mrr": float(sum(all_mrr) / max(1, len(all_mrr)))
    }
    return avg_loss, metrics



def train_baseline(
    lags: int = 12,
    hidden_dim: int = 32,
    K: int = 2,
    epochs: int = 10,
    lr: float = 1e-3,
    target_channel: int = 0,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = TGBDatasetLoader(name="tgbl-wiki")

    num_node_feats = dataset.num_node_feats
    num_edge_feats = dataset.num_edge_feats

    model = DCRNNLinkPredictor(in_channels=num_node_feats, d_emb=hidden_dim, d_edge_attr=num_edge_feats, K=K).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    snapshot_count = 500
    warmup_steps = 1000
    # split by time
    train_end = int(0.7 * snapshot_count)
    val_end = int(0.85 * snapshot_count)

    transform = T.RandomLinkSplit(
        num_val = 0,
        num_test = 0,
        add_negative_train_samples = True,
        neg_sampling_ratio = 1.0,
    )

    for epoch in range(1, epochs + 1):
        model.train()

        snapshots = dataset.get_snapshots(warmup_steps=warmup_steps)

        count = 0
        total_loss = 0.0

        for t in range(0, train_end):
            snapshot = next(snapshots)
            
            # negative edge sample
            snapshot, _, _ = transform(snapshot)

            snapshot.to(device)

            edge_label_index = snapshot.edge_label_index
            edge_label = snapshot.edge_label

            logits, H = model(snapshot, edge_label_index)

            loss = bce(logits, edge_label)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            count += 1
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, count)

        val_loss, val_m = eval_epoch(
            model, snapshots, device, transform,
            t_start=train_end, t_end=val_end,
            k_eval=50,
        )

        test_loss, test_m = eval_epoch(
            model, snapshots, device, transform,
            t_start=val_end, t_end=snapshot_count,
            k_eval=50,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val(AUC={val_m['auc']:.3f}, F1={val_m['f1']:.3f}, P@50={val_m['p@k']:.3f}, MRR={val_m['mrr']:.3f}; "
            f"test_loss={test_loss:.6f} "
            f"test(AUC={test_m['auc']:.3f}, F1={test_m['f1']:.3f}, P@50={test_m['p@k']:.3f},  MRR={test_m['mrr']:.3f})"
        )

    return model


if __name__ == "__main__":
    train_baseline(
        lags=12,
        hidden_dim=32,
        K=2,
        epochs=10,
        lr=1e-3,
        target_channel=0,
    )

