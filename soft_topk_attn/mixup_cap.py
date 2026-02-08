import torch
import torch.nn.functional as F
import random


class MixupWithMemory:
    def __init__(self, num_classes, d_emb, device="cpu", memory_max_size=100):
        self.num_classes = num_classes
        self.d_emb = d_emb
        self.device = device

        # memory buffers
        self.memory_context = None
        self.memory_query = None
        self.memory_target = None

        # ===== CHANGED =====
        # Cap the memory size to avoid unbounded growth (OOM / SIGKILL)
        self.memory_max_size = memory_max_size
        # ===================

    def update_memory(self, context, query, target):
        """
        context: [B, d]
        query:   [B, d]
        target:  [B]
        """
        if self.memory_context is None:
            self.memory_context = context.detach()
            self.memory_query = query.detach()
            self.memory_target = target.detach()
        else:
            self.memory_context = torch.cat([self.memory_context, context.detach()], dim=0)
            self.memory_query = torch.cat([self.memory_query, query.detach()], dim=0)
            self.memory_target = torch.cat([self.memory_target, target.detach()], dim=0)

        # ===== CHANGED =====
        # Keep only the most recent memory_max_size samples (FIFO)
        if self.memory_context.size(0) > self.memory_max_size:
            self.memory_context = self.memory_context[-self.memory_max_size:]
            self.memory_query = self.memory_query[-self.memory_max_size:]
            self.memory_target = self.memory_target[-self.memory_max_size:]
        # ===================

    def get_mixup_samples(self, context, query, target):
        """
        context: [B, d]
        query:   [B, d]
        target:  [B]
        """
        self.update_memory(context, query, target)

        if self.memory_context is None or self.memory_context.size(0) < 2:
            return context, target, target, target, query

        B = context.size(0)
        mem_size = self.memory_context.size(0)

        idx_i = torch.randint(0, mem_size, (B,), device=self.device)
        idx_j = torch.randint(0, mem_size, (B,), device=self.device)

        ctx_i = self.memory_context[idx_i]
        ctx_j = self.memory_context[idx_j]
        qry_i = self.memory_query[idx_i]
        qry_j = self.memory_query[idx_j]
        tgt_i = self.memory_target[idx_i]
        tgt_j = self.memory_target[idx_j]

        lam = torch.rand(B, 1, device=self.device)

        mix_ctx = lam * ctx_i + (1 - lam) * ctx_j
        mix_qry = lam * qry_i + (1 - lam) * qry_j

        tgt_i_one_hot = F.one_hot(tgt_i, num_classes=self.num_classes).float()
        tgt_j_one_hot = F.one_hot(tgt_j, num_classes=self.num_classes).float()
        mix_tgt = lam * tgt_i_one_hot + (1 - lam) * tgt_j_one_hot

        return mix_ctx, mix_tgt, tgt_i, tgt_j, mix_qry
