#   Train PILGRIM on the PEMS BAY Node-level prediction task
#   Use tSNE to visualize the learned node embeddings for the last snapshot after training.


from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_topk_attn.data.pems_bay import load_pemsbay_from_npy
from soft_topk_attn.models.attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.diversity_loss_f import diversity_loss
from soft_topk_attn.models.mixup import MixupWithMemory
from soft_topk_attn.models.metric_loss import MetricLoss


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# attention, mixup, metric loss settings
INIT_K_FRAC = 0.05
K_MIN = 0.01
K_MAX = 0.5
TAU = 0.05
BETA_DIV = 0.1
BETA_METRIC = 0.01
BETA_MIXUP = 1.0

# batching (to reduce memory)
ANCHORS_PER_SNAPSHOT_TRAIN = 32
ANCHORS_PER_SNAPSHOT_EVAL = -1
MAX_EVAL_ANCHORS = 256

# split
TRAIN_RATIO = 0.7  # 20% train, 80% test
TOTAL_RATIO = 0.025  # total % data for train+test
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


def ce_margin_score(logits2: torch.Tensor) -> torch.Tensor:
    return logits2[..., 1] - logits2[..., 0]


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
    
@torch.inference_mode()
def visualize(
    model: DCRNNEncoder,
    snapshots,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    lags: int,
    threshold: float,
    target_channel: int,
    horizon: int,
    t_vis: int,
    device: torch.device,
    output_path: str = "pems_pilgrim_tsne.png",
):
    model.eval()
    x_vis = _to_tensor(snapshots[t_vis].x, device=device, dtype=torch.float)
    if horizon == 1:
        y_vis = _to_tensor(snapshots[t_vis].y, device=device, dtype=torch.float)
    else:
        y_vis = _to_tensor(snapshots[min(t_vis + 2, len(snapshots) - 1)].y, device=device, dtype=torch.float)

    X_vis = x_to_batched_sequence(x_vis, lags=lags)

    with torch.inference_mode():
        H_vis = model(X_vis, edge_index, edge_weight)

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
    
    # Step 2: tSNE 16 -> 2
    print("Step 2: Running tSNE to reduce 16 -> 2 dimensions...")
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, len(emb_16d)-1))
    emb_2d = tsne.fit_transform(emb_16d)
    print(f"tSNE completed. 2D embeddings shape: {emb_2d.shape}")

    # Save embeddings to file
    embeddings_file = output_path.replace('.png', '_embeddings.npz')
    np.savez(embeddings_file,
             emb_original=emb_np,
             emb_pca_16d=emb_16d,
             emb_tsne_2d=emb_2d,
             labels=labels_vis,
             t_vis=t_vis)
    print(f"Saved embeddings to {embeddings_file}")

    plt.figure(figsize=(8, 6))
    for lab, color, name in [(0.0, "#1f77b4", "label=0"), (1.0, "#d62728", "label=1")]:
        mask = labels_vis == lab
        if mask.any():
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.7, c=color, label=name)
    plt.title(f"PEMS {MODEL_NAME} (w/ Attention) tSNE embeddings (PCA->tSNE: 32->16->2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved tSNE plot to {output_path}")


def train_and_visualize_one_run():
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
    train_end = max(1, int(TRAIN_RATIO * T))
    test_start = train_end
    test_end = T

    h_idx = 0 if HORIZON == 1 else 1

    print(f"=== ({MODEL_NAME}) single-horizon CE: HORIZON={HORIZON} ===")
    print(f"Split: train [0, {train_end}) ({train_end}/{T}={train_end/T:.1%}), test [{test_start}, {test_end})")

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); attention.train(); query_embed.train(); head.train()

        total = 0.0
        steps = 0

        max_h = 3
        t_end_eff = min(train_end, len(snapshots) - (max_h - 1))

        for t in range(0, t_end_eff):
            print(f"{t} / {T}", end="\r")
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

                emb_q = query_embed(torch.tensor(h_idx, device=device))

                context, attn, soft_mask, scores, Q, K, theta, k = attention(
                    emb_q=emb_q,
                    emb_a=emb_a,
                    emb_nodes=emb_nodes,
                    node_mask=node_mask,
                    tau=TAU,
                    return_intermediates=True,
                )

                rep = emb_a +emb_q + context
                logits2 = head(rep)

                # Node-level label: check if node's future value > threshold
                if y_future.dim() > 1:
                    val = y_future[anchor_idx, TARGET_CHANNEL]
                else:
                    val = y_future[anchor_idx]
                y_class = (val > THRESHOLD).long()

                loss_t = loss_t + ce(logits2, y_class)

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


        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
        )

    # Visualize the last training snapshot after all training completes.
    t_vis = max(0, min(train_end - 1, len(snapshots) - 1))
    visualize(
        model=encoder,
        snapshots=snapshots,
        edge_index=edge_index,
        edge_weight=edge_weight,
        lags=LAGS,
        threshold=THRESHOLD,
        target_channel=TARGET_CHANNEL,
        horizon=HORIZON,
        t_vis=t_vis,
        device=device,
        output_path="pems_pilgrim_tsne.png",
    )


if __name__ == "__main__":
    train_and_visualize_one_run()
