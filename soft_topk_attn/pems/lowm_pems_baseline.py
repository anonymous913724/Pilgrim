import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import DCRNN

from soft_topk_attn.data.pems_bay import load_pemsbay_from_npy
from soft_topk_attn.models.metrics_bin import (
    binary_auc_from_logits,
    binary_f1_from_logits,
    precision_at_k_from_logits
)

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

EPOCHS = 10
LR = 1e-3

K_EVAL = 50
F1_THR = 0.5

# split
TRAIN_RATIO = 0.7 # 20% train, 80% test
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


@torch.inference_mode()
def eval_epoch(
    model: nn.Module,
    snapshots,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    threshold: float,
    target_channel: int,
    horizon: int,
    lags: int,
    num_features: int,
    device: torch.device,
    t_start: int,
    t_end: int,
    k_eval: int = 50,
    f1_thr: float = 0.5,
):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    count = 0

    all_labels = []
    all_logits = []
    p_at_k = []

    max_h = 3
    t_end_eff = min(t_end, len(snapshots) - (max_h - 1))

    # For +3 label: use y at t+2 (because y is "next step")
    for t in range(t_start, t_end_eff):
        print(f"Eval {t} / {t_end_eff}", end="\r")
        x = torch.tensor(snapshots[t].x, device=device, dtype=torch.float)   # [N, lags*F]
        if horizon == 1:
            y_future = torch.tensor(snapshots[t].y, device=device, dtype=torch.float)
        else:
            y_future = torch.tensor(snapshots[t + 2].y, device=device, dtype=torch.float)

        X = x_flat_to_seq(x, lags=lags, num_features=num_features)
        logits = model(X, edge_index, edge_weight).squeeze()                           # [N]

        val = y_future[:, target_channel] if y_future.dim() > 1 else y_future

        labels = torch.stack([(val > threshold).float()], dim=-1).squeeze()  # [N]

        loss = bce(logits, labels)
        total_loss += float(loss.item())
        count += 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        p_at_k.append(precision_at_k_from_logits(logits, labels, k=k_eval))

    avg_loss = total_loss / max(1, count)

    if count == 0:
        metrics = {
            "auc": 0.5, "f1": 0.0, "p@k": 0.0,
        }
        return avg_loss, metrics

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = {
        "auc": binary_auc_from_logits(logits, labels),
        "f1":  binary_f1_from_logits(logits, labels, thr=f1_thr),
        "p@k": float(sum(p_at_k) / max(1, len(p_at_k))),
    }
    return avg_loss, metrics



def train_baseline(
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
        test_loss, test_m = eval_epoch(
            model, snapshots, edge_index, edge_weight,
            horizon=HORIZON, threshold=threshold, target_channel=target_channel, lags=lags, num_features=num_features, device=device,
            t_start=train_end, t_end=test_end,
            k_eval=K_EVAL, f1_thr=F1_THR
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} "
            f"test(AUC={test_m['auc']:.3f}, F1={test_m['f1']:.3f}, P@50={test_m['p@k']:.3f})"
        )

    return model


if __name__ == "__main__":

    train_baseline(
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

