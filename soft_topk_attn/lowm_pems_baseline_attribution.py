"""
Attribution analysis for DCRNN baseline model using Embedding-level Integrated Gradients.

Computes deletion curves by:
1. Using embedding-level IG to identify top-K important nodes
2. Progressively removing these nodes by masking their embeddings
3. Tracking F1 score degradation
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from pems_bay import load_pemsbay_from_npy
from integrated_gradients import (
    compute_integrated_gradients,
    compute_node_importance_scores,
)
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
HORIZON = 1              # 1 or 3
LAGS = 12
D_EMB = 32
DCRNN_K = 2

THRESHOLD = 60.0
TARGET_CHANNEL = 0
ADJ_THRESHOLD = 0.0

EPOCHS = 1
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# attribution settings
DELETION_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# batching
ANCHORS_PER_SNAPSHOT_EVAL = 64
MAX_EVAL_ANCHORS = 512

# split
TRAIN_RATIO = 0.95
TOTAL_RATIO = 0.025
ADJ_PATH = "data/pems_adj_mat.npy"
VALUES_PATH = "data/pems_node_values.npy"


def _to_tensor(x, device, dtype=torch.float):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def x_flat_to_seq(x_flat: torch.Tensor, lags: int, num_features: int) -> torch.Tensor:
    """
    x_flat: [N, lags*F] -> X: [1, lags, N, F]
    """
    N = x_flat.size(0)
    X = x_flat.view(N, lags, num_features).permute(1, 0, 2).unsqueeze(0)  # [1,L,N,F]
    return X

class DCRNNBinaryBaseline(nn.Module):
    """
    DCRNN-based binary classification baseline model for traffic prediction that can use different DGNN backbones.
    """
    def __init__(self, in_channels: int, d_emb: int, K: int = 2):
        super().__init__()
        name = str(MODEL_NAME).strip().upper()

        if name == "DCRNN":
            from torch_geometric_temporal.nn.recurrent import DCRNN as _DCRNN
            self.cell = _DCRNN(in_channels=in_channels, out_channels=d_emb, K=K)

        elif name == "SEHTGNN":
            from SEHTGNN import SEHTGNN as _SEHTGNN
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

        self.head = nn.Linear(d_emb, 1)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
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
        logits = self.head(H)  # [N, 1]
        return logits
    
    def get_embeddings(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """Extract node embeddings H from the GNN (before classification head)"""
        # X: [1, L, N, F]
        
        if hasattr(self.cell, "reset_state"):
            self.cell.reset_state()

        H = None
        for t in range(X.size(1)):
            x_t = X[0, t]  # [N, F]
            try:
                H = self.cell(x_t, edge_index, edge_weight, H)
            except TypeError:
                H = self.cell(x_t, edge_index, edge_weight)
        return H  # [N, d_emb]


def extract_embeddings(
    baseline_model: DCRNNBinaryBaseline,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    lags: int,
    num_features: int,
) -> torch.Tensor:
    """
    Extract node embeddings from the baseline model (before classification head).
    
    Args:
        baseline_model: The GNN model
        node_features: [N, lags*num_features] flattened temporal features
        edge_index: [2, num_edges]
        edge_weight: [num_edges] or None
        lags: Number of time lags
        num_features: Number of features per timestep
        
    Returns:
        embeddings: [N, d_emb] node embeddings
    """
    # Reshape to temporal format
    X = x_flat_to_seq(node_features, lags=lags, num_features=num_features)
    
    # Extract embeddings (before head)
    embeddings = baseline_model.get_embeddings(X, edge_index, edge_weight)
    
    return embeddings


@torch.inference_mode()
def compute_deletion_curve(
    baseline_model: DCRNNBinaryBaseline,
    node_features_batch: torch.Tensor,  # [B, N, lags*F]
    labels_batch: torch.Tensor,          # [B]
    anchor_indices_batch: torch.Tensor,  # [B] anchor node indices
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    attributions_batch: List[torch.Tensor],  # [B] of [N, d_emb]
    removal_fracs: List[float],
    lags: int,
    num_features: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute F1 scores after progressively deleting top-K important nodes based on embedding attributions.
    
    Args:
        baseline_model: Original baseline model for predictions
        node_features_batch: [B, N, lags*F] batch of node features
        labels_batch: [B] batch of labels
        anchor_indices_batch: [B] anchor node indices for each sample
        attributions_batch: [B] list of attribution tensors [N, d_emb]
        removal_fracs: List of deletion fractions [0.0, 0.2, ...]
        
    Returns:
        (fractions, f1_scores, area_under_curve)
    """
    device = node_features_batch.device
    batch_size = node_features_batch.size(0)
    num_nodes = 17
    
    # Prepare fractions
    fracs_t = torch.as_tensor(removal_fracs, device=device, dtype=torch.float)
    if fracs_t.numel() == 0:
        fracs_t = torch.tensor([0.0], device=device)
    if float(fracs_t.min()) > 0.0:
        fracs_t = torch.cat([torch.zeros(1, device=device), fracs_t])
    fracs_t = torch.clamp(fracs_t, 0.0, 1.0)
    fracs_t, _ = torch.sort(fracs_t)
    
    f1_scores = []
    
    # For each deletion fraction
    for frac in fracs_t.tolist():
        all_preds = []
        all_labels = []
        
        # Process each anchor in the batch
        for i in range(batch_size):
            node_features_i = node_features_batch[i]  # [N, lags*F]
            attribution_i = attributions_batch[i]  # [N, d_emb]
            label_i = labels_batch[i]
            anchor_idx = anchor_indices_batch[i].item()
            
            # Rank nodes by importance
            node_scores = compute_node_importance_scores(attribution_i, aggregation='l2')
            sorted_indices = torch.argsort(node_scores, descending=True)
            
            # Determine how many to delete
            num_to_delete = int(round(frac * num_nodes))
            num_to_delete = min(max(num_to_delete, 0), num_nodes)
            
            # Get embeddings and mask out deleted nodes
            embeddings = extract_embeddings(
                baseline_model, node_features_i, edge_index, edge_weight, lags, num_features
            )
            
            # Zero out embeddings of deleted nodes
            embeddings_masked = embeddings.clone()
            if num_to_delete > 0:
                nodes_to_delete = sorted_indices[:num_to_delete]
                embeddings_masked[nodes_to_delete] = 0.0
            
            # Predict using masked embeddings
            with torch.enable_grad():
                logits = baseline_model.head(embeddings_masked)  # [N, 1]
            
            # Get prediction for this anchor
            logits_anchor = logits[anchor_idx].squeeze()  # scalar
            pred = (torch.sigmoid(logits_anchor) > 0.5).float()
            
            all_preds.append(pred)
            all_labels.append(label_i)
        
        # Compute F1 across batch
        all_preds_t = torch.stack(all_preds)  # [B]
        all_labels_t = torch.stack(all_labels)  # [B]
        
        tp = ((all_preds_t == 1) & (all_labels_t == 1)).sum().float()
        fp = ((all_preds_t == 1) & (all_labels_t == 0)).sum().float()
        fn = ((all_preds_t == 0) & (all_labels_t == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        f1_scores.append(f1)
    
    scores_t = torch.tensor(f1_scores, device=device)
    auc = torch.trapz(scores_t, fracs_t).item()
    
    return fracs_t, scores_t, auc


@torch.no_grad()
def compute_sufficiency(
    head: nn.Module,
    embeddings: torch.Tensor,
    attributions: torch.Tensor,
    k: int,
) -> float:
    """
    Compute sufficiency: |f(G) - f(G_Sk)| using embedding-level IG importance.

    f(G) uses the mean of all node embeddings. f(G_Sk) uses the mean of top-k nodes.
    """
    num_nodes = embeddings.size(0)
    if num_nodes == 0:
        return 0.0

    k = max(1, min(int(k), num_nodes))

    full_agg = embeddings.mean(dim=0)
    full_prob = torch.sigmoid(head(full_agg).squeeze())

    node_scores = compute_node_importance_scores(attributions, aggregation="l2")
    topk = torch.argsort(node_scores, descending=True)[:k]
    induced_agg = embeddings[topk].mean(dim=0)
    induced_prob = torch.sigmoid(head(induced_agg).squeeze())

    sufficiency = abs(full_prob - induced_prob)

    return sufficiency


def eval_epoch_with_attribution(
    baseline_model: DCRNNBinaryBaseline,
    snapshots,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    lags: int,
    num_features: int,
    horizon: int,
    threshold: float,
    target_channel: int,
    device: torch.device,
    t_start: int,
    t_end: int,
    anchors_per_snapshot: int,
    max_eval_anchors: int,
    deletion_fracs: Optional[List[float]] = None,
    compute_deletion: bool = True,
    compute_sufficiency: bool = True,
) -> Tuple[float, dict]:
    """
    Evaluation with embedding-level IG deletion curves on baseline model.
    Explains only the classification head, not the full GNN pipeline.
    """
    baseline_model.eval()
    bce = nn.BCEWithLogitsLoss()
    
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
        
        x = _to_tensor(snapshots[t].x, device=device, dtype=torch.float)  # [N, lags*F]
        if horizon == 1:
            y_future = _to_tensor(snapshots[t].y, device=device, dtype=torch.float)
        else:
            y_future = _to_tensor(snapshots[t + 2].y, device=device, dtype=torch.float)
        
        X = x_flat_to_seq(x, lags=lags, num_features=num_features)
        logits = baseline_model(X, edge_index, edge_weight)  # [N, 1]
        
        # Compute BCE loss
        v_future = y_future[:, target_channel] if y_future.dim() > 1 else y_future
        labels = (v_future > threshold).float()  # [N]
        loss = bce(logits.squeeze(), labels)
        total_loss += float(loss.item())
        steps += 1
        
        # Sample anchors for this snapshot
        num_nodes = x.size(0)
        candidates = torch.arange(num_nodes, device=device)
        
        if max_eval_anchors is not None and max_eval_anchors > 0 and candidates.numel() > max_eval_anchors:
            perm = torch.randperm(candidates.numel(), device=device)
            candidates = candidates[perm[:max_eval_anchors]]
        
        if anchors_per_snapshot >= 0 and anchors_per_snapshot < candidates.numel():
            perm = torch.randperm(candidates.numel(), device=device)
            anchor_indices = candidates[perm[:anchors_per_snapshot]]
        else:
            anchor_indices = candidates
        
        # Extract embeddings once for all anchors (reuse GNN computation)
        embeddings = extract_embeddings(
            baseline_model, x, edge_index, edge_weight, lags, num_features
        )  # [N, d_emb]
        
        # Collect data for deletion curve computation
        snapshot_labels = []
        snapshot_features = []
        snapshot_anchor_indices = []
        snapshot_attributions = []
        
        for anchor_idx in anchor_indices.tolist():
            # Define classification head for this anchor
            def classification_head(agg_embedding):
                # agg_embedding: [d_emb] -> logits for anchor node
                # We want to explain: how does each node's embedding contribute to anchor's prediction?
                # For simplicity, we'll aggregate all embeddings and predict
                return baseline_model.head(agg_embedding).squeeze()
            
            # Compute embedding-level IG attributions
            # This explains which nodes' embeddings contribute most to the anchor's prediction
            with torch.enable_grad():
                attributions = compute_integrated_gradients(
                    embeddings=embeddings,
                    classification_head=classification_head,
                    baseline_type='zero',
                    steps=20,  # Reduced steps for speed
                    target_class=None,  # Binary classification (single output)
                    attention_weights=None,  # Uniform aggregation
                )  # [N, d_emb]
            
            # Store for deletion curve
            snapshot_labels.append(labels[anchor_idx].detach())
            snapshot_features.append(x.detach())
            snapshot_anchor_indices.append(torch.tensor(anchor_idx, device=device))
            snapshot_attributions.append(attributions.detach())

            if compute_sufficiency:
                suff = compute_sufficiency(
                    head=baseline_model.head,
                    embeddings=embeddings,
                    attributions=attributions,
                    k=K_EVAL,
                )
                suff_list.append(suff)

        if compute_deletion and len(snapshot_labels) > 0:
            # Stack data for batch processing
            labels_batch = torch.stack(snapshot_labels)  # [B]
            features_batch = torch.stack(snapshot_features)  # [B, N, lags*F]
            anchor_indices_batch = torch.stack(snapshot_anchor_indices)  # [B]
            
            # Compute deletion curve
            fracs_t, del_scores, del_auc = compute_deletion_curve(
                baseline_model=baseline_model,
                node_features_batch=features_batch,
                labels_batch=labels_batch,
                anchor_indices_batch=anchor_indices_batch,
                edge_index=edge_index,
                edge_weight=edge_weight,
                attributions_batch=snapshot_attributions,
                removal_fracs=deletion_fracs,
                lags=lags,
                num_features=num_features,
            )
            
            del_auc_list.append(del_auc)
            del_curves.append(del_scores.detach().cpu())
            del_fracs_ref = fracs_t.detach().cpu()
    
    avg_loss = total_loss / max(1, steps)
    
    metrics = {
        "deletion_auc": float(sum(del_auc_list) / max(1, len(del_auc_list))) if compute_deletion else 0.0,
        "deletion_curve": torch.stack(del_curves).mean(dim=0).tolist() if del_curves else [],
        "deletion_fracs": del_fracs_ref.tolist() if del_fracs_ref is not None else [],
        "sufficiency": float(sum(suff_list) / max(1, len(suff_list))) if compute_sufficiency else 0.0,
    }
    
    return avg_loss, metrics


def train_baseline_with_attribution():
    """
    Train baseline model and evaluate with GB-IG deletion curves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    dataset = load_pemsbay_from_npy(ADJ_PATH, VALUES_PATH, lags=LAGS, horizon=1, adj_threshold=ADJ_THRESHOLD)
    snapshots = list(dataset)
    snapshots = snapshots[:int(TOTAL_RATIO * len(snapshots))]
    
    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long, device=device)
    edge_weight = torch.tensor(dataset.edge_weight, dtype=torch.float, device=device)
    
    # Get dimensions
    x0 = torch.tensor(snapshots[0].x)
    assert x0.dim() == 2 and x0.shape[1] % LAGS == 0
    num_features = int(x0.shape[1] // LAGS)
    num_nodes = int(x0.shape[0])
    
    print(f"Dataset: {num_nodes} nodes, {num_features} features per step, {LAGS} steps")
    
    # Create model
    model = DCRNNBinaryBaseline(in_channels=num_features, d_emb=D_EMB, K=DCRNN_K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    
    # Split data
    T = len(snapshots)
    train_end = max(1, int(TRAIN_RATIO * T))
    test_start = train_end
    test_end = T
    
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end/T:.1%}), test [{test_start}, {test_end})")
    print(f"Training with horizon={HORIZON} for {EPOCHS} epochs")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0
        
        max_h = 3
        t_end_eff = min(train_end, len(snapshots) - (max_h - 1))
        
        for t in range(0, t_end_eff):
            print(f"Epoch {epoch} train {t} / {t_end_eff}", end="\r")
            
            x = _to_tensor(snapshots[t].x, device=device, dtype=torch.float)
            if HORIZON == 1:
                y_future = _to_tensor(snapshots[t].y, device=device, dtype=torch.float)
            else:
                y_future = _to_tensor(snapshots[t + 2].y, device=device, dtype=torch.float)
            
            X = x_flat_to_seq(x, lags=LAGS, num_features=num_features)
            logits = model(X, edge_index, edge_weight)  # [N, 1]
            
            v_future = y_future[:, TARGET_CHANNEL] if y_future.dim() > 1 else y_future
            labels = (v_future > THRESHOLD).float()
            
            loss = bce(logits.squeeze(), labels)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            total_loss += float(loss.item())
            count += 1
        
        train_loss = total_loss / max(1, count)
        
        # Evaluate with attribution (needs gradients for IG)
        test_loss, test_m = eval_epoch_with_attribution(
            baseline_model=model,
            snapshots=snapshots,
            edge_index=edge_index,
            edge_weight=edge_weight,
            lags=LAGS,
            num_features=num_features,
            horizon=HORIZON,
            threshold=THRESHOLD,
            target_channel=TARGET_CHANNEL,
            device=device,
            t_start=test_start,
            t_end=test_end,
            anchors_per_snapshot=ANCHORS_PER_SNAPSHOT_EVAL,
            max_eval_anchors=MAX_EVAL_ANCHORS,
            deletion_fracs=DELETION_FRACS,
            compute_deletion=False
        )
        
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} "
            f"deletion_auc={test_m['deletion_auc']:.3f} "
            f"suff={test_m['sufficiency']:.3f}"
        )
    
    # Plot deletion curve
    if len(test_m["deletion_fracs"]) > 0 and len(test_m["deletion_curve"]) > 0:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        plt.plot(test_m["deletion_fracs"], test_m["deletion_curve"], marker="o", linewidth=2)
        plt.xlabel("Fraction of top-K nodes deleted")
        plt.ylabel("F1 score")
        plt.title(f"Baseline Model Deletion Curve (Horizon={HORIZON})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"baseline_deletion_curve_h{HORIZON}.png", dpi=150)
        print(f"Saved deletion curve to baseline_deletion_curve_h{HORIZON}.png")
        plt.close()

        np.savez(
            f"baseline_deletion_curve_h{HORIZON}.npz",
            fracs=np.array(test_m["deletion_fracs"]),
            scores=np.array(test_m["deletion_curve"]),
            auc=test_m["deletion_auc"],
        )
        print(f"Saved deletion curve data to baseline_deletion_curve_h{HORIZON}.npz")


if __name__ == "__main__":
    train_baseline_with_attribution()
