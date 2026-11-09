"""Allocator-page access for both dense hybrid storage and planar K/V views.

Tensor-page mode packs/unpacks the actual backend view; it never assumes K and
V for a physical block occupy one adjacent raw-byte range. Works on CPU in
tests and on the actual CUDA tensor in workers.
"""
from dataclasses import dataclass
from math import prod


@dataclass
class PageStorage:
    tensor: object
    page_size: int
    block_axis: int | None = None
    factor: int = 1

    def __post_init__(self):
        if self.page_size < 1 or self.factor < 1:
            raise ValueError("invalid allocator page size/factor")
        if self.block_axis is None:
            if self.tensor.numel() % self.page_size:
                raise ValueError("raw storage is not a whole number of allocator pages")
        else:
            if not 0 <= self.block_axis < self.tensor.ndim:
                raise ValueError("invalid block axis")
            if self.tensor.shape[self.block_axis] % self.factor:
                raise ValueError("kernel blocks do not align to allocator pages")
            shape = list(self.tensor.shape)
            shape[self.block_axis] = self.factor
            if prod(shape) * self.tensor.element_size() != self.page_size:
                raise ValueError("logical K/V slice does not match allocator page bytes")

    @property
    def num_pages(self):
        if self.block_axis is None:
            return self.tensor.numel() // self.page_size
        return self.tensor.shape[self.block_axis] // self.factor

    @property
    def layout(self):
        if self.block_axis is None:
            return {}  # Preserve existing dense-hybrid artifact layout hashes.
        shape = list(self.tensor.shape)
        shape[self.block_axis] = self.factor
        return {"copy_mode": "logical_tensor_page", "page_shape": shape,
                "block_axis": self.block_axis, "factor": self.factor,
                "dtype": str(self.tensor.dtype)}

    def _slice(self, block_id):
        if type(block_id) is not int or not 0 <= block_id < self.num_pages:
            raise ValueError("physical block ID outside allocator storage")
        if self.block_axis is None:
            return self.tensor.narrow(0, block_id * self.page_size, self.page_size)
        return self.tensor.narrow(self.block_axis, block_id * self.factor, self.factor)

    def read(self, block_id):
        import torch
        return self._slice(block_id).contiguous().view(torch.uint8).reshape(-1)

    def write(self, block_id, byte_page):
        import torch
        if byte_page.dtype != torch.uint8 or byte_page.numel() != self.page_size:
            raise ValueError("invalid allocator byte page")
        target = self._slice(block_id)
        source = byte_page.contiguous().view(target.dtype).reshape(target.shape)
        target.copy_(source, non_blocking=False)
