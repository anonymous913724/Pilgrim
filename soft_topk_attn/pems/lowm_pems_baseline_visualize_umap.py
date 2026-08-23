#   Train DGNN (baseline) on the PEMS BAY Node-level prediction task
#   Use UMAP to visualize the learned node embeddings for the last snapshot after training.

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import DCRNN

from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from soft_topk_attn.data.pems_bay import load_pemsbay_from_npy

# =========================
# SETTINGS
# =========================
HORIZON = 1              # set to 1 or 3 (predict 1-step or 3-step future)
LAGS = 12
D_EMB = 64
DCRNN_K = 2

THRESHOLD = 65.0
TARGET_CHANNEL = 0
ADJ_THRESHOLD = 0.0

EPOCHS = 2
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# split
TRAIN_RATIO = 0.9 # 90% train, 10% test
TOTAL_RATIO = 0.025 # total % data for train+test
ADJ_PATH = "data/pems_adj_mat.npy"
VALUES_PATH = "data/pems_node_values.npy"



def x_flat_to_seq(x_flat: torch.Tensor, lags: int, num_features: int) -> torch.Tensor:
    """
    x_flat: [N, lags*F] -> X: [1, lags, N, F]
    """
    N = x_flat.size(0)
    X = x_flat.view(N, lags, num_features).permute(1, 0, 2).unsqueeze(0)  # [1,L,N,F]
    return X


class DCRNNBinaryBaseline(nn.Module):
    """
    Use DCRNN step-by-step over the lags window.
    """
    def __init__(self, in_channels: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = DCRNN(in_channels=in_channels, out_channels=hidden_dim, K=K)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        # X: [1, L, N, F]
        assert X.dim() == 4 and X.size(0) == 1
        L = X.size(1)
        H = None
        for t in range(L):
            x_t = X[0, t]  # [N,F]
            try:
                H = self.cell(x_t, edge_index, edge_weight, H)
            except TypeError:
                H = self.cell(x_t, edge_index, edge_weight)
        # H: [N, hidden]
        logits = self.head(H)
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

        return H


@torch.inference_mode()
def visualize(
    model: DCRNNBinaryBaseline,
    snapshots,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    lags: int,
    num_features: int,
    threshold: float,
    target_channel: int,
    horizon: int,
    t_vis: int,
    device: torch.device,
    output_path: str = "pems_baseline_umap.png",
):
    model.eval()
    x_vis = torch.tensor(snapshots[t_vis].x, device=device, dtype=torch.float)
    if horizon == 1:
        y_vis = torch.tensor(snapshots[t_vis].y, device=device, dtype=torch.float)
    else:
        y_vis = torch.tensor(snapshots[min(t_vis + 2, len(snapshots) - 1)].y, device=device, dtype=torch.float)

    X_vis = x_flat_to_seq(x_vis, lags=lags, num_features=num_features)

    with torch.inference_mode():
        H_vis = model.get_embeddings(X_vis, edge_index, edge_weight)

    val_vis = y_vis[:, target_channel] if y_vis.dim() > 1 else y_vis
    labels_vis = (val_vis > threshold).float().detach().cpu().numpy()

    emb_np = H_vis.detach().cpu().numpy()
    print(f"Extracted node embeddings H_vis with shape {emb_np.shape} for t={t_vis}")
    print(f"Label distribution for visualization snapshot: {labels_vis.sum()} positive, {len(labels_vis) - labels_vis.sum()} negative")
    
    # Step 1: PCA 32 -> 16
    print("Step 1: Applying PCA to reduce 32 -> 16 dimensions...")
    pca = PCA(n_components=16, random_state=42)
    emb_16d = pca.fit_transform(emb_np)
    print(f"PCA completed. 16D embeddings shape: {emb_16d.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Step 2: UMAP 16 -> 2
    print("Step 2: Running UMAP to reduce 16 -> 2 dimensions...")
    umap_reducer = umap.UMAP(n_components=2, min_dist=0.1, random_state=42)
    emb_2d = umap_reducer.fit_transform(emb_16d)
    print(f"UMAP completed. 2D embeddings shape: {emb_2d.shape}")

    # Save embeddings to file
    embeddings_file = output_path.replace('.png', '_embeddings.npz')
    np.savez(embeddings_file, 
             emb_original=emb_np,
             emb_pca_16d=emb_16d,
             emb_umap_2d=emb_2d,
             labels=labels_vis,
             t_vis=t_vis)
    print(f"Saved embeddings to {embeddings_file}")

    plt.figure(figsize=(8, 6))
    for lab, color, name in [(0.0, "#1f77b4", "label=0"), (1.0, "#d62728", "label=1")]:
        mask = labels_vis == lab
        if mask.any():
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.7, c=color, label=name)
    plt.title("PEMS Baseline UMAP embeddings (PCA->UMAP: 32->16->2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved UMAP plot to {output_path}")



def train_and_visualize_baseline(
    adj_path: str,
    values_path: str,
    lags: int = 12,
    hidden_dim: int = 32,
    K: int = 2,
    epochs: int = 10,
    lr: float = 1e-3,
    threshold: float = 60.0,
    target_channel: int = 0,
    adj_threshold: float = 0.0,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = load_pemsbay_from_npy(adj_path, values_path, lags=lags, horizon=1, adj_threshold=adj_threshold)
    snapshots = list(dataset)
    snapshots = snapshots[:int(TOTAL_RATIO * len(snapshots))]

    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long, device=device)
    edge_weight = torch.tensor(dataset.edge_weight, dtype=torch.float, device=device)

    # infer F from snapshot.x shape [N, lags*F]
    x0 = torch.tensor(snapshots[0].x)
    N, D = x0.shape
    assert D % lags == 0
    num_features = D // lags

    model = DCRNNBinaryBaseline(in_channels=num_features, hidden_dim=hidden_dim, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    T = len(snapshots)
    train_end = max(1, int(TRAIN_RATIO * T))
    test_start = train_end
    test_end = T

    print(f"=== Baseline (DCRNN)) single-horizon CE: HORIZON={HORIZON} ===")
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end/T:.1%}), test [{test_start}, {test_end})")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        max_h = 3

        t_end_eff = min(train_end, len(snapshots) - (max_h - 1))

        for t in range(0, t_end_eff):
            print(f"{t} / {T}", end="\r")
            x = torch.tensor(snapshots[t].x, device=device, dtype=torch.float)
            if HORIZON == 1:
                y_future = torch.tensor(snapshots[t].y, device=device, dtype=torch.float)
            else:
                y_future = torch.tensor(snapshots[t + 2].y, device=device, dtype=torch.float)

            X = x_flat_to_seq(x, lags=lags, num_features=num_features)
            logits = model(X, edge_index, edge_weight)                           # [N,2]

            val = y_future[:, target_channel] if y_future.dim() > 1 else y_future

            labels = torch.stack([(val > threshold).float()], dim=-1)  # [N,1]

            loss = bce(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            count += 1

        train_loss = total_loss / max(1, count)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
        )
    # After training, visualize node embeddings with UMAP on the last training snapshot.
    t_vis = max(0, min(train_end - 1, len(snapshots) - 1))
    visualize(
        model=model,
        snapshots=snapshots,
        edge_index=edge_index,
        edge_weight=edge_weight,
        lags=lags,
        num_features=num_features,
        threshold=threshold,
        target_channel=target_channel,
        horizon=HORIZON,
        t_vis=t_vis,
        device=device,
        output_path="pems_baseline_umap_4.png",
    )

    return model


if __name__ == "__main__":
    train_and_visualize_baseline(
        adj_path=ADJ_PATH,
        values_path=VALUES_PATH,
        lags=LAGS,
        hidden_dim=D_EMB,
        K=DCRNN_K,
        epochs=EPOCHS,
        lr=LR,
        threshold=THRESHOLD,
        target_channel=TARGET_CHANNEL,
        adj_threshold=ADJ_THRESHOLD,
    )
