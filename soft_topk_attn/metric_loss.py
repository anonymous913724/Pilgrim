import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricLoss(nn.Module):
    def __init__(self, num_classes, d_emb, code_size, beta=1, lamb=0.5, device=None):
        super(MetricLoss, self).__init__()
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # better initialization? kaiming? xaiver?
        self.codebook = nn.Parameter(torch.randn(num_classes, code_size)).to(self.device)
        self.evidence_lin = nn.Linear(d_emb, code_size).to(self.device)
        self.num_classes = num_classes
        self.beta = beta
        self.lamb = lamb

    def forward(
        self,
        evidence: torch.Tensor,
        evidence_mixup: torch.Tensor,
        task: torch.Tensor,
        task_mixup: torch.Tensor,
        target: torch.Tensor,
        target_i: torch.Tensor,
        target_j: torch.Tensor
    ):
        orig_loss = self._orig_loss(evidence, task, target)

        mix_loss = self._mix_loss(evidence_mixup,
                                  task_mixup,
                                  target_i,
                                  target_j)
        
        return orig_loss + mix_loss
    
    def _simplex_loss(self):
        G = self.codebook @ self.codebook.t() # Gram matrix of codebook
        G = G - self.beta
        G = F.relu(G)

        # sum strictly upper triangle (i < j)
        return torch.triu(G, diagonal=1).sum()
     
    def _orig_loss(self, evidence, task, target):
        if evidence.ndim != task.ndim:
            evidence = evidence.unsqueeze(0)

        # pool evidence and task
        evidence += task
        
        evidence_proj = self.evidence_lin(evidence)
        target = target.to(torch.int64)
        codes = self.codebook[target]
        x = evidence_proj - codes
        x = x.norm(dim=-1, p=2)
        x = x + self._simplex_loss()
        return x
    
    def _mix_loss(self, evidence_mixup, task, target_i, target_j):

        # pool evidence after mixup and task
        evidence_mixup += task

        mixup_proj = self.evidence_lin(evidence_mixup)
        
        # perform codebook mixup on targets
        target_i = target_i.to(torch.int64)
        target_j = target_j.to(torch.int64)

        codes_i = self.codebook[target_i]
        codes_i = self.lamb *  codes_i
        codes_j = self.codebook[target_j]
        codes_j = (1 - self.lamb) * codes_j


        x = mixup_proj - (codes_i + codes_j)
        x = x.norm(dim=-1, p=2)

        return x
