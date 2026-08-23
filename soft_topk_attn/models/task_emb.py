import torch
from sentence_transformers import SentenceTransformer
from typing import Union, Sequence
import torch.nn as nn


class task_embedding(nn.Module):
    """
    Encodes a task specified by (text_description, task_type, output_type, temporal_scope)
    using SentenceTransformer('all-MiniLM-L6-v2') into a 384-d vector, then projects to 32-d.
    """

    def __init__(
        self,
        text_description: Union[str, Sequence[str]],
        task_type: Union[str, Sequence[str]],
        output_type: Union[str, Sequence[str]],
        temporal_scope: Union[str, Sequence[str]],
        pretrained: str = "all-MiniLM-L6-v2",
        output_dim: int = 32,
    ):
        super().__init__()
        self.embedder = SentenceTransformer(pretrained)
        self._embed_dim = 384
        self.linear = nn.Linear(self._embed_dim, output_dim)

        # Normalize inputs to lists of strings
        def _to_list(x):
            if isinstance(x, str):
                return [x]
            if isinstance(x, Sequence):
                return list(x)
            raise TypeError("Inputs must be str or Sequence[str]")

        td = _to_list(text_description)
        tt = _to_list(task_type)
        ot = _to_list(output_type)
        ts = _to_list(temporal_scope)

        if not (len(td) == len(tt) == len(ot) == len(ts)):
            raise ValueError("All input sequences must have the same length")

        # Concatenate fields per item
        concatenated = [f"{a} | {b} | {c} | {d}" for a, b, c, d in zip(td, tt, ot, ts)]

        # Encode to a torch tensor of shape (batch, _embed_dim)
        emb = self.embedder.encode(concatenated, convert_to_tensor=True)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if emb.size(-1) != self._embed_dim:
            raise ValueError(f"Expected embed dim {self._embed_dim}, got {emb.size(-1)}")
        self.task_emb = emb

    def forward(self) -> torch.Tensor:
        """
        Returns: tensor of shape (batch_size, output_dim)
        """
        emb = self.task_emb.to(next(self.linear.parameters()).device)
        out = self.linear(emb)
        return out