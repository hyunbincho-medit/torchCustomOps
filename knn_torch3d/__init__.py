"""KNN operations with torch custom ops"""

from collections import namedtuple
from typing import Union

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# C++ 확장 로드
try:
    from . import _core
    # torch ops 로드
    torch.ops.load_library(_core.__file__)
except ImportError as e:
    raise ImportError(f"Cannot load knn_torch3d extension: {e}")

_KNN = namedtuple("KNN", "dists idx knn")


class _knn_points(Function):
    """Torch autograd Function wrapper for KNN C++/CUDA implementations."""

    @staticmethod
    def forward(
        ctx,
        p1,
        p2,
        lengths1,
        lengths2,
        K,
        version,
        norm: int = 2,
        return_sorted: bool = True,
    ):
        """K-Nearest neighbors on point clouds."""
        if not ((norm == 1) or (norm == 2)):
            raise ValueError("Support for 1 or 2 norm.")

        # torch.ops를 통해 custom op 호출
        idx, dists = torch.ops.knn_torch3d.knn_points_idx(
            p1, p2, lengths1, lengths2, norm, K, version
        )

        # sort KNN in ascending order if K > 1
        if K > 1 and return_sorted:
            if lengths2.min() < K:
                P1 = p1.shape[1]
                mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
                mask = mask[:, None].expand(-1, P1, -1)
                dists[mask] = float("inf")
                dists, sort_idx = dists.sort(dim=2)
                dists[mask] = 0
            else:
                dists, sort_idx = dists.sort(dim=2)
            idx = idx.gather(2, sort_idx)

        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        ctx.norm = norm
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        norm = ctx.norm
        
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()
            
        grad_p1, grad_p2 = torch.ops.knn_torch3d.knn_points_backward(
            p1, p2, lengths1, lengths2, idx, norm, grad_dists
        )
        return grad_p1, grad_p2, None, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D)
        p2: Tensor of shape (N, P2, D)
        lengths1: LongTensor of shape (N,) or None
        lengths2: LongTensor of shape (N,) or None
        norm: Integer (1 for L1, 2 for L2)
        K: Number of nearest neighbors
        version: KNN implementation version
        return_nn: Whether to return nearest neighbors
        return_sorted: Whether to sort by distance

    Returns:
        Named tuple with dists, idx, knn
    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, norm, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    """
    Helper function for knn that allows indexing a tensor x with indices idx.

    Args:
        x: Tensor of shape (N, M, U)
        idx: LongTensor of shape (N, L, K)
        lengths: LongTensor of shape (N,) or None

    Returns:
        x_out: Tensor of shape (N, L, K, U)
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    needs_mask = lengths.min() < K
    if needs_mask:
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out


__all__ = ['knn_points', 'knn_gather', '_KNN']