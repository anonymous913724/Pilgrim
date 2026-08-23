import torch


def diversity_loss(attn: torch.Tensor, emb_nodes: torch.Tensor, k: int, node_mask: torch.Tensor = None,) -> torch.Tensor:
    """
    attn:      [N]
    emb_nodes: [N, d_in]
    k: hard top-k for diversity
    node_mask: optional [N] bool/0-1
    Returns a scalar diversity loss encouraging selected nodes to be dissimilar
    (penalizes pairwise similarity of weighted embeddings).

    sum_{i!=j} (alpha_i*z_i) dot (alpha_j*z_j) = sum_{i!=j} alpha_i*alpha_j*(z_i dot z_j)
    if s = sum alpha_i*z_i, then s dot s = sum_{i,j} alpha_i*alpha_j(z_i dot z_j)
    Split this into diagonal and off-diagonal parts:  sum_{i,j} = sum_{i!=j} + sum_{i}
    We can get: sum_{i!=j} alpha_i*alpha_j*(z_i dot z_j) = s dot s - sum_i alpha_i^2 * ||z_i||^2
    So sum_{i!=j} (alpha_i*z_i) dot (alpha_j*z_j) = ||sum_i alpha_i*z_i||^2 - sum_i alpha_i^2 * ||z_i||^2
    """
    if node_mask is None:
        valid = torch.ones_like(attn, dtype=torch.bool)
    else:
        valid = node_mask.bool().view(-1)
    num_valid = int(valid.sum().item())
    if num_valid <= 1 or k <= 1:
        return emb_nodes.sum() * 0.0  # scalar zero on correct device/dtype
    k_eff = min(int(k), num_valid)
    
    # Top-k among valid nodes
    attn_valid = attn[valid]  # [N_valid]
    topk_attn, topk_pos = torch.topk(attn_valid, k_eff, largest=True, sorted=False)

    valid_idx = torch.nonzero(valid, as_tuple=False).view(-1) # [N_valid]
    topk_idx = valid_idx[topk_pos]                            # [k_eff]

    # Compute u_i = attn_i * z_i on selected nodes (NO soft_mask)
    z = emb_nodes[topk_idx]                                   # [k_eff, d_in]
    w = topk_attn.to(dtype=emb_nodes.dtype).unsqueeze(-1)     # [k_eff, 1]
    u = w * z						      # [k_eff, d_in] u_i=alpha_i*z_i

    sum_u = torch.sum(u, dim=0)				      # [d_in]
    term_all = torch.dot(sum_u, sum_u)                        # ||sum_i alpha_i*z_i||^2
    term_diag = torch.sum(u * u)                              # sum_i alpha_i^2 * ||z_i||^2

    return term_all - term_diag # sum_{i!=j} (alpha_i*z_i) dot (alpha_j*z_j)
    
