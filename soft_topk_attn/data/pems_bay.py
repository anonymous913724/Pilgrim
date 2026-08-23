import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch


def adj_to_edge_index_weight(adj: np.ndarray, threshold: float = 0.0):
    """
    adj: [N, N]
    Returns:
      edge_index: [2, E] int64
      edge_weight: [E] float32
    """
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], f"adj must be [N,N], got {adj.shape}"
    src, dst = np.where(adj > threshold)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_weight = adj[src, dst].astype(np.float32)
    return edge_index, edge_weight


def load_pemsbay_from_npy(
    adj_path: str,
    values_path: str,
    lags: int = 12,
    horizon: int = 1,
    adj_threshold: float = 0.0,
    target_feature: int = 0,
):
    adj = np.load(adj_path)
    values = np.load(values_path)

    N_adj = adj.shape[0]
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], f"adj must be [N,N], got {adj.shape}"

    # --- Normalize values to [T, N, F] using N_adj ---
    if values.ndim == 2:
        # values: [T,N] or [N,T]
        if values.shape[0] == N_adj and values.shape[1] != N_adj:
            # [N,T] -> [T,N]
            values = values.T
        elif values.shape[1] == N_adj and values.shape[0] != N_adj:
            # already [T,N]
            pass
        elif values.shape[0] == N_adj and values.shape[1] == N_adj:
            # ambiguous square (unlikely); treat as [T,N]
            pass
        else:
            raise ValueError(f"Cannot align values {values.shape} with adj N={N_adj}")
        values = values[:, :, None]  # [T,N,1]

    elif values.ndim == 3:
        # values: [T,N,F] or [N,T,F]
        if values.shape[1] == N_adj:
            # [T,N,F]
            pass
        elif values.shape[0] == N_adj:
            # [N,T,F] -> [T,N,F]
            values = np.transpose(values, (1, 0, 2))
        else:
            raise ValueError(f"Cannot align values {values.shape} with adj N={N_adj}")

    else:
        raise ValueError(f"Unsupported node_values shape: {values.shape}")
    # --- end normalize ---

    T, N, F = values.shape
    assert N == N_adj, f"Mismatch after normalize: N(values)={N} vs N(adj)={N_adj}"

    edge_index, edge_weight = adj_to_edge_index_weight(adj, threshold=adj_threshold)

    features = []
    targets = []
    for t in range(lags, T - horizon + 1):
        x_win = values[t - lags: t]       # [lags, N, F]
        y_tgt = values[t + horizon - 1]   # [N, F]

        x_flat = np.transpose(x_win, (1, 0, 2)).reshape(N, -1).astype(np.float32)
        y_val = y_tgt.astype(np.float32)

        features.append(x_flat)
        targets.append(y_val)

    return StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)

