from torch_geometric.data import Data
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_layer import QKOnlySoftTopKAttention
from soft_topk_attn.models.diversity_loss_f import diversity_loss


def example_loss_func(emb_q: torch.Tensor, emb_a: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    target = emb_q + emb_a
    return F.mse_loss(context, target)

def run_on_snapshot_sequence(
    snapshots,                       # list[Data]
    encoder_x: nn.Module,            # encodes data.x -> emb_nodes
    attention: QKOnlySoftTopKAttention,
    optimizer: torch.optim.Optimizer,
    beta: float = 0.1,
    device: Optional[torch.device] = None,
    tau: Optional[float] = None,
    query_encoder: Optional[nn.Module] = None,
    train: bool = True,
) -> float:
    """
    Debug code
    Train/eval over a sequence of fake snapshots (no batching).
    Returns average loss over snapshots.
    Use MSE as head part loss
    Use nn.Linear to encode the input nodes and query
    """
    if device is None:
        device = torch.device("cpu")

    encoder_x.to(device)
    attention.to(device)
    if query_encoder is not None:
        query_encoder.to(device)

    if train:
        encoder_x.train()
        attention.train()
        if query_encoder is not None:
            query_encoder.train()
    else:
        encoder_x.eval()
        attention.eval()
        if query_encoder is not None:
            query_encoder.eval()

    total_loss = 0.0
    count = 0

    for data in snapshots:
        data = data.to(device)

        # 1) Encode node embeddings
        emb_nodes = encoder_x(data.x)                 # [N, d_emb]

        # 2) Get anchor embedding (use task-node embedding directly for debugging)
        anchor_idx = int(data.anchor_idx) if hasattr(data, "anchor_idx") else 0
        emb_a = emb_nodes[anchor_idx]                 # [d_emb]

        # 3) Get query embedding
        if hasattr(data, "query_feat"):
            q_feat = data.query_feat
            emb_q = query_encoder(q_feat) if query_encoder is not None else q_feat
        elif hasattr(data, "query_idx"):
            q_idx = int(data.query_idx)
            emb_q = emb_nodes[q_idx]
        else:
            # fallback: use anchor as query (just for a runnable example)
            emb_q = emb_a

        # Optional node validity mask (if you have it). Otherwise None.
        node_mask = data.node_mask if hasattr(data, "node_mask") else None

        # 4) Attention -> context (soft top-k mask inside)
        context, attn, soft_mask = attention(
            emb_q=emb_q,
            emb_a=emb_a,
            emb_nodes=emb_nodes,
            node_mask=node_mask,
            tau=tau,
        )

        # 5) Compute losses
        base = example_loss_func(emb_q, emb_a, context)
        div = diversity_loss(attn, soft_mask, emb_nodes)
        loss = base + beta * div

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        count += 1

    return total_loss / max(count, 1)

class SimpleEncoder(nn.Module):
    def __init__(self, x_dim: int, d_emb: int):
        super().__init__()
        self.lin = nn.Linear(x_dim, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_dim = 16
d_emb = 32
q_dim = 8

encoder_x = SimpleEncoder(x_dim, d_emb)
query_encoder = nn.Linear(q_dim, d_emb)

#Newton version
attention = QKOnlySoftTopKAttention(
    d_in=d_emb,
    d_out=d_emb,
    tau=0.05,
    init_k_frac=0.05, #0.2
    normalize_qk=False,
    newton_iters=15,
    newton_damping=1.0,
)

#OLD version -- Using bisection method, not Newton
# attention = QKOnlySoftTopKAttention(
#     d_in=d_emb,
#     d_out=d_emb,       # often set d_out == d_emb for simplicity
#     tau=0.05,
#     init_k_frac=0.2,        # initial k ≈ 0.2 * N_valid
#     normalize_qk=False,
# )

params = list(encoder_x.parameters()) + list(query_encoder.parameters()) + list(attention.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# Fake snapshot sequence (replace with your PyG-Temporal snapshots)
snapshots = []
for t in range(10):
    N = 50
    data = Data(
        x=torch.randn(N, x_dim),
        anchor_idx=torch.tensor(0),
        query_feat=torch.randn(q_dim),
    )
    snapshots.append(data)

for epoch in range(1, 11):
    avg_loss = run_on_snapshot_sequence(
        snapshots=snapshots,
        encoder_x=encoder_x,
        query_encoder=query_encoder,
        attention=attention,
        optimizer=optimizer,
        beta=0.1,
        device=device,
        tau=0.05,
        train=True,
    )
    with torch.no_grad():
        data0 = snapshots[0].to(device)

        emb_nodes0 = encoder_x(data0.x)
        emb_a0 = emb_nodes0[int(data0.anchor_idx)]

        if hasattr(data0, "query_feat"):
            emb_q0 = query_encoder(data0.query_feat.to(device))
        else:
            emb_q0 = emb_a0

        out = attention(
            emb_q=emb_q0,
            emb_a=emb_a0,
            emb_nodes=emb_nodes0,
            node_mask=(data0.node_mask if hasattr(data0, "node_mask") else None),
            tau=0.05,
            return_intermediates=True,
        )

        context0, attn0, soft_mask0, scores0, Q0, K0, theta0, k0 = out

        mask_sum = soft_mask0.sum().item()
        k_val = k0.item()
        k_frac = torch.sigmoid(attention.k_logit).item()

        print(
            f"  [check] k_frac={k_frac:.4f}, "
            f"k={k_val:.3f}, "
            f"mask_sum={mask_sum:.3f}, "
            f"diff={mask_sum - k_val:+.3e}"
        )

    # with torch.no_grad():
    #     data0 = snapshots[0].to(device)
    #     emb_nodes0 = encoder_x(data0.x)
    #     emb_a0 = emb_nodes0[int(data0.anchor_idx)]
    #     emb_q0 = query_encoder(data0.query_feat.to(device))
    #     out = attention(
    #         emb_q=emb_q0,
    #         emb_a=emb_a0,
    #         emb_nodes=emb_nodes0,
    #         node_mask=(data0.node_mask if hasattr(data0, "node_mask") else None),
    #         tau=0.05,
    #         return_intermediates=True,
    #     )
    #     context0, alpha0, mask0, scores0, Q0, K0, theta0, k0 = out
    #     print(f"  solved theta={float(theta0.cpu()):.6f}, learned k={float(k0.cpu()):.2f}")

    #non IFT version
    #print(f"epoch={epoch} avg_loss={avg_loss:.6f} theta={float(attention.theta.detach().cpu()):.4f}")
    #IFT version
    print(f"epoch={epoch} avg_loss={avg_loss:.6f} k_frac={float(torch.sigmoid(attention.k_logit).detach().cpu()):.4f}")

    # OLD version -- Using bisection method, not Newton
    # print(
    #     f"epoch={epoch} avg_loss={avg_loss:.6f} "
    #     f"k_frac={float(torch.sigmoid(attention.k_logit).detach().cpu()):.4f}"
    # )

