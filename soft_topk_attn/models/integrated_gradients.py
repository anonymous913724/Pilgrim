# Embedding-level Integrated Gradients (IG) for node importance attribution
#
import torch
from typing import Optional, Tuple, Callable, Literal


def compute_integrated_gradients(
    embeddings: torch.Tensor,
    classification_head: Callable[[torch.Tensor], torch.Tensor],
    baseline: Optional[torch.Tensor] = None,
    baseline_type: Literal['zero', 'mean', 'custom'] = 'zero',
    steps: int = 50,
    target_class: Optional[int] = None,
    attention_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute integrated gradients for node embeddings w.r.t classification head.
    
    Args:
        embeddings: [num_nodes, embed_dim]
        classification_head: Function that takes aggregated embedding and returns logits
                           Should accept [embed_dim] tensor and return [num_classes] tensor
        baseline: Custom baseline embeddings [num_nodes, embed_dim]. Used if baseline_type='custom'
        baseline_type: Type of baseline to use ('zero', 'mean', 'custom')
        steps: Number of interpolation steps for Riemann approximation
        target_class: If specified, compute IG for this specific class. Otherwise use max logit.
        attention_weights: Optional attention weights [num_nodes] for weighted aggregation
        
    Returns:
        attributions: Attribution scores [num_nodes, embed_dim]
    """
    num_nodes, _ = embeddings.shape
    device = embeddings.device
    
    # Create baseline embeddings
    if baseline_type == 'zero':
        baseline_embeddings = torch.zeros_like(embeddings)
    elif baseline_type == 'mean':
        mean_embedding = embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
        baseline_embeddings = mean_embedding.repeat(num_nodes, 1)
    elif baseline_type == 'custom':
        if baseline is None:
            raise ValueError("baseline must be provided when baseline_type='custom'")
        baseline_embeddings = baseline
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")
    
    # Initialize attributions buffer
    attributions = torch.zeros_like(embeddings)
    
    # Compute path difference
    path_diff = embeddings - baseline_embeddings  # [num_nodes, embed_dim]
    
    # Default attention weights (uniform)
    if attention_weights is None:
        attention_weights = torch.ones(num_nodes, device=device) / num_nodes
    else:
        # Normalize attention weights
        attention_weights = attention_weights / attention_weights.sum()
    
    # Riemann sum approximation of the integral
    for step in range(steps):
        # Interpolation coefficient: alpha from 0 to 1
        alpha = (step + 0.5) / steps 
        
        # Interpolated embeddings: z(alpha) = baseline + alpha * (z - baseline)
        interpolated_embeddings = baseline_embeddings + alpha * path_diff
        interpolated_embeddings = interpolated_embeddings.requires_grad_(True)
        
        # Aggregate embeddings: z_agg = sum_i (w_i * z_i)
        # [num_nodes, embed_dim] -> [embed_dim]
        aggregated = (interpolated_embeddings * attention_weights.unsqueeze(1)).sum(dim=0)
        
        # Forward pass through classification head
        logits = classification_head(aggregated)  # [num_classes] or scalar
        
        # Select target output
        if target_class is not None:
            if logits.dim() == 0:
                output = logits  # Scalar output
            else:
                output = logits[target_class]
        else:
            # Use max logit (predicted class)
            if logits.dim() == 0:
                output = logits
            else:
                output = logits.max()
        
        # Compute gradients w.r.t interpolated embeddings
        grads = torch.autograd.grad(
            outputs=output,
            inputs=interpolated_embeddings,
            retain_graph=False,
            create_graph=False
        )[0]  # [num_nodes, embed_dim]
        
        # Accumulate gradients
        attributions += grads
    
    # multiply by path difference and divide by steps to approximate integral
    attributions = attributions * path_diff / steps
    
    return attributions


def compute_node_importance_scores(
    attributions: torch.Tensor,
    aggregation: Literal['l2', 'l1', 'mean', 'sum', 'max'] = 'l2'
) -> torch.Tensor:
    """
    Aggregate attribution scores across embedding dimensions to get per-node importance.
    
    Args:
        attributions: Attribution scores [num_nodes, embed_dim]
        aggregation: Method to aggregate across embedding dimension
        
    Returns:
        node_scores: Importance score for each node [num_nodes]
    """
    if aggregation == 'l2':
        # L2 norm of attributions
        node_scores = torch.norm(attributions, p=2, dim=1)
    elif aggregation == 'l1':
        # L1 norm (sum of absolute values)
        node_scores = torch.norm(attributions, p=1, dim=1)
    elif aggregation == 'mean':
        # Mean absolute attribution
        node_scores = torch.abs(attributions).mean(dim=1)
    elif aggregation == 'sum':
        # Sum of attributions (signed)
        node_scores = attributions.sum(dim=1)
    elif aggregation == 'max':
        # Maximum absolute attribution
        node_scores = torch.abs(attributions).max(dim=1)[0]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return node_scores


def get_top_k_nodes(
    attributions: torch.Tensor,
    k: int,
    aggregation: Literal['l2', 'l1', 'mean', 'sum', 'max'] = 'l2',
    return_scores: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract top-K most important nodes based on attribution scores.
    
    Args:
        attributions: Attribution scores [num_nodes, embed_dim]
        k: Number of top nodes to return
        aggregation: Method to aggregate across embedding dimension
        return_scores: Whether to return importance scores
        
    Returns:
        top_k_indices: Indices of top-K nodes [k]
        top_k_scores: Importance scores (if return_scores=True) [k]
    """
    # Compute per-node importance scores
    node_scores = compute_node_importance_scores(attributions, aggregation)
    
    # Get top-K
    k = min(k, len(node_scores))  # Handle k > num_nodes
    top_k_scores, top_k_indices = torch.topk(node_scores, k, largest=True)
    
    if return_scores:
        return top_k_indices, top_k_scores
    else:
        return top_k_indices, None