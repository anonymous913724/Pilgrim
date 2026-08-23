import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QKOnlySoftTopKAttention(nn.Module):
    """
    QK-only attention:
      Q = (emb_q + emb_a) W_Q
      K_i = emb_i W_K
      score_i = (Q · K_i) / sqrt(d_out)
      attn = softmax(score)

    Soft top-k mask:
      mask_i = sigmoid((attn_i - theta)/tau)

    k is learnable (as a fraction of valid nodes)
    theta is solved in forward so that sum(mask) = k

    """

    def __init__(
        self,
        d_in: int, #input dimension
        d_out: int, #output dimension
        tau: float = 0.1,
        init_k_frac: float = 0.005,     # initial k approximate to init_k_frac * N_valid
        k_min: float = 0.001,
        k_max: float = 0.01,
        normalize_qk: bool = False,
        # Newton method settings
        newton_iters: int = 15,
        newton_damping: float = 1.0,
        theta_margin: float = 10.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.d_out = d_out
        self.tau = float(tau)
        self.normalize_qk = normalize_qk

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)

        # Learnable k fraction in (0,1): k = sigmoid(k_logit) * N_valid
        # the k is calculated by following:
        # self.k_logit = nn.Parameter(...)
        # k_frac = torch.sigmoid(self.k_logit)  # in (0,1)
        # k = k_frac * N_valid
        # This guarantees:
        # k_frac belong to (0, 1)
        # k belong to (0, N_valid)
        # Smooth gradients
        
        # --- bounded k_frac in [k_min, k_max] ---
        assert 0.0 < float(k_min) < float(k_max) <= 1.0
        self.k_min = float(k_min)
        self.k_max = float(k_max)

        # clip init_k_frac into (k_min, k_max) with tiny margin
        init_k_frac = float(init_k_frac)
        init_k_frac = max(self.k_min + 1e-6, min(self.k_max - 1e-6, init_k_frac))

        # map init_k_frac -> u in (0,1):  k_frac = k_min + (k_max - k_min) * u
        init_u = (init_k_frac - self.k_min) / (self.k_max - self.k_min)
        init_u = max(1e-6, min(1.0 - 1e-6, float(init_u)))

        # raw logit parameter (unconstrained)
        init_logit = math.log(init_u) - math.log(1.0 - init_u)
        self.k_logit = nn.Parameter(torch.tensor(init_logit))

        self.newton_iters = int(newton_iters)
        self.newton_damping = float(newton_damping)
        self.theta_margin = float(theta_margin)
        self.eps = float(eps)

    def _solve_theta_newton(
        self,
        attn: torch.Tensor,           # [N]
        k: torch.Tensor,              # scalar tensor
        tau_t: torch.Tensor,          # scalar tensor
        valid: torch.Tensor,          # [N] bool
    ) -> torch.Tensor:
        """
        Solve theta for: sum sigmoid((attn - theta)/tau) = k
        using damped Newton-Raphson, with clamping to a safe bracket.
        """
        attn_v = attn[valid]                      # [N_valid]
        N_valid = attn_v.numel()

        # Clamp k to feasible range [0, N_valid]
        k = torch.clamp(k, 0.0, float(N_valid))

        # Bracket theta (keeps Newton stable)
        lo = attn_v.min() - self.theta_margin * tau_t
        hi = attn_v.max() + self.theta_margin * tau_t

        # Initialize theta (mean is usually fine)
        theta = attn_v.mean()

        for _ in range(self.newton_iters):
            z = (attn_v - theta) / tau_t
            m = torch.sigmoid(z)                  # [N_valid]
            f = m.sum() - k                       # scalar

            # f'(theta) = d/dtheta sum sigmoid((a-theta)/tau)
            #          = - (1/tau) * sum m*(1-m)
            g = m * (1.0 - m)
            fp = -(g.sum() / tau_t)               # negative scalar

            # Avoid division blow-up
            fp = torch.where(fp.abs() < self.eps, fp.sign() * self.eps, fp)

            step = f / fp
            theta = theta - self.newton_damping * step

            # clamp to bracket
            theta = torch.clamp(theta, lo, hi)

        return theta

    def forward(
        self,
        emb_q: torch.Tensor,                      # [d_in]
        emb_a: torch.Tensor,                      # [d_in]
        emb_nodes: torch.Tensor,                  # [N, d_in]
        node_mask: Optional[torch.Tensor] = None, # [N] bool/0-1 optional
        tau: Optional[float] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          context: [d_in]
          attn:    [N]
          soft_mask: [N]
        """
        if tau is None:
            tau = self.tau
        tau_t = torch.tensor(float(tau), device=emb_nodes.device, dtype=emb_nodes.dtype)

        # Q = (emb_q + emb_a) W_Q
        qa = emb_q + emb_a
        Q = self.W_q(qa)                          # [d_out]

        # K = emb_nodes W_K
        K = self.W_k(emb_nodes)                   # [N, d_out]

        if self.normalize_qk:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)

        # scores = (K @ Q) / sqrt(d_out)
        scores = torch.matmul(K, Q) / math.sqrt(self.d_out)  # [N]

        if node_mask is not None:
            scores = scores.masked_fill(~node_mask.bool(), float("-inf")) #~ == not for boolean

        attn = F.softmax(scores, dim=0)           # [N]

        # valid nodes for theta/k computation
        if node_mask is None:
            valid = torch.ones_like(attn, dtype=torch.bool)
        else:
            valid = node_mask.bool()
        N_valid = int(valid.sum().item()) #total number of valid nodes

        # learn k dynamically as a fraction of N_valid
        u = torch.sigmoid(self.k_logit)  # in (0,1)
        k_frac = self.k_min + (self.k_max - self.k_min) * u  # in (k_min, k_max)
        k = k_frac * float(N_valid)


        # solve theta using Newton
        theta = self._solve_theta_newton(attn, k, tau_t, valid)

        # soft mask using solved theta
        soft_mask = torch.sigmoid((attn - theta) / tau_t)    # [N]
        if node_mask is not None:
            soft_mask = soft_mask * node_mask.to(dtype=soft_mask.dtype)

        w = (soft_mask * attn).unsqueeze(-1)      # [N, 1]
        context = torch.sum(w * emb_nodes, dim=0) # [d_in] # z

        if return_intermediates:
            return context, attn, soft_mask, scores, Q, K, theta, k
        return context, attn, soft_mask # context (evidence), attention, soft top-k mask



