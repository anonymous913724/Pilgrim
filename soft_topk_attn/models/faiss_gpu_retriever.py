# helper module for FAISS-GPU candidate retrieval (faiss-gpu==1.7.2).
#
# Goal: build a GPU index once per snapshot over K = W_k(emb_nodes),
# then for each anchor query, retrieve top-m candidate node ids on GPU.
#
# Important notes for faiss-gpu==1.7.2:
# - Always use float32 for FAISS add/search.
# - Torch tensor I/O through faiss.contrib.torch_utils may or may not be available depending on your build.
#   If you want to guarantee the torch-GPU fastpath (no numpy fallback), set require_torch_gpu=True.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import faiss

_TORCH_UTILS_OK = False
try:
    import faiss.contrib.torch_utils  # noqa: F401
    _TORCH_UTILS_OK = True
except Exception:
    _TORCH_UTILS_OK = False


def assert_torch_gpu_fastpath(require: bool = True) -> None:
    """Tiny check: hard-fail if torch-GPU tensor I/O path is not available."""
    if require and (not _TORCH_UTILS_OK):
        raise RuntimeError(
            "FAISS torch-GPU fastpath is NOT available: cannot import faiss.contrib.torch_utils. "
            "Install/compile faiss-gpu with torch_utils support, or set FAISS_REQUIRE_TORCH_GPU=False."
        )


@dataclass
class FaissGpuIndexState:
    index: object
    valid_idx: torch.Tensor  # [Nv] (CUDA long), maps FAISS row -> original node id
    d: int
    metric: str
    normalize: bool


class FaissGpuRetriever:
    """FAISS GPU retriever (IndexFlatIP/IndexFlatL2) with optional strict torch-GPU enforcement."""

    def __init__(self, device: torch.device, metric: str = "ip"):
        if device.type != "cuda":
            raise ValueError("FaissGpuRetriever requires CUDA device.")
        self.device = device
        self.metric = str(metric).lower().strip()
        if self.metric not in ("ip", "l2"):
            raise ValueError("metric must be 'ip' or 'l2'")
        self.res = faiss.StandardGpuResources()
        self.state: Optional[FaissGpuIndexState] = None
        self._index = None  # keep a persistent GPU index to avoid re-alloc each build()

    def _make_index(self, d: int):
        if self.metric == "ip":
            cpu_index = faiss.IndexFlatIP(d)
        else:
            cpu_index = faiss.IndexFlatL2(d)
        # put index on GPU 0 (same device as PyTorch CUDA:0 usage)
        return faiss.index_cpu_to_gpu(self.res, 0, cpu_index)

    @torch.no_grad()
    def build(
        self,
        K_valid: torch.Tensor,   # [Nv, d]
        valid_idx: torch.Tensor, # [Nv]
        normalize: bool = False,
        require_torch_gpu: bool = True,  # strict fastpath check
    ) -> None:
        """Build/refresh the GPU index over the provided key vectors."""
        assert_torch_gpu_fastpath(require=require_torch_gpu)

        if K_valid.ndim != 2:
            raise ValueError("K_valid must be 2D [Nv, d].")
        if valid_idx.ndim != 1:
            raise ValueError("valid_idx must be 1D [Nv].")
        if K_valid.size(0) != valid_idx.size(0):
            raise ValueError("K_valid and valid_idx must have same first dimension.")

        d = int(K_valid.size(1))
        # reuse the same GPU index object if d doesn't change
        if (self.state is not None) and (int(self.state.d) == d):
            index = self.state.index
        else:
            index = self._make_index(d)
        index.reset()

        # faiss expects float32
        if K_valid.dtype != torch.float32:
            K_valid = K_valid.float()
        if normalize:
            K_valid = torch.nn.functional.normalize(K_valid, dim=-1)

        K_valid = K_valid.contiguous()
        valid_idx = valid_idx.contiguous()

        # If strict mode: already asserted torch_utils import; also require CUDA tensors.
        if require_torch_gpu:
            if not K_valid.is_cuda:
                raise RuntimeError("require_torch_gpu=True but K_valid is not CUDA tensor.")
            index.add(K_valid)  # torch-GPU path
        else:
            # fallback: numpy add
            if K_valid.is_cuda and _TORCH_UTILS_OK:
                index.add(K_valid)
            else:
                index.add(K_valid.detach().cpu().numpy())

        if not valid_idx.is_cuda:
            valid_idx = valid_idx.to(device=self.device)

        self.state = FaissGpuIndexState(
            index=index,
            valid_idx=valid_idx.to(dtype=torch.long),
            d=d,
            metric=self.metric,
            normalize=bool(normalize),
        )

    @torch.no_grad()
    def search(
        self,
        Q: torch.Tensor,  # [d] or [B,d]
        topm: int,
        normalize: bool = False,
        require_torch_gpu: bool = True,  # strict fastpath check
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (candidate_node_ids, scores/distances)."""
        assert_torch_gpu_fastpath(require=require_torch_gpu)

        if self.state is None:
            raise RuntimeError("FAISS index not built yet. Call build() first.")
        st = self.state

        if Q.ndim == 1:
            Q = Q.unsqueeze(0)
        if Q.ndim != 2:
            raise ValueError("Q must be [d] or [B,d].")

        if Q.dtype != torch.float32:
            Q = Q.float()
        if normalize:
            Q = torch.nn.functional.normalize(Q, dim=-1)
        Q = Q.contiguous()

        if require_torch_gpu:
            if not Q.is_cuda:
                raise RuntimeError("require_torch_gpu=True but Q is not CUDA tensor.")
            D, I = st.index.search(Q, int(topm))   # torch-GPU path
            cand = st.valid_idx[I]
            return cand, D

        # fallback mode
        if Q.is_cuda and _TORCH_UTILS_OK:
            D, I = st.index.search(Q, int(topm))
            cand = st.valid_idx[I]
            return cand, D

        D_np, I_np = st.index.search(Q.detach().cpu().numpy(), int(topm))
        I = torch.from_numpy(I_np).to(device=self.device, dtype=torch.long)
        D = torch.from_numpy(D_np).to(device=self.device, dtype=torch.float32)
        cand = st.valid_idx[I]
        return cand, D
