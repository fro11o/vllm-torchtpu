# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared attention-metadata, cache-slot, RoPE, and replay helpers for DeepSeek-V4.1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch import Tensor
from torch.tpu import pallas
from vllm import forward_context

from vllm_torchtpu.kernels.experimental.deepseek_v41.rope import rope
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

from .cache import DsV41CacheMixin, PagedCacheRows

_KV_ROPE_OP_NAME = "pallas::dsv41_kv_rope"
_kv_rope_op_cache: dict[str, Any] = {}


def _kv_rope_jax(
    inputs: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
) -> jax.Array:
    return rope(inputs, positions, cos_sin_cache)


def _fake_kv_rope(
    inputs: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
) -> Tensor:
    del positions, cos_sin_cache
    return inputs.new_empty(inputs.shape)


def get_kv_rope_op() -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return cached ``pallas::dsv41_kv_rope`` op for trailing-channel RoPE.

    Shapes: ``inputs=(T, D)`` or ``(T, H, D)`` bf16 with ``D % 128 == 0``,
    ``positions=(T,)`` int32, ``cos_sin_cache=(max_pos, rotary_dim)`` f32.
    """
    mesh = get_pallas_op_mesh()
    # Avoid colliding in PyTorch's global custom-op registry across mesh modes:
    # registering a multi-device mesh under the same name would clobber mesh=None.
    op_name = (
        _KV_ROPE_OP_NAME
        if mesh is None
        else f"{_KV_ROPE_OP_NAME}_mesh_{'_'.join(mesh.axis_names)}_{mesh.size}"
    )
    op = _kv_rope_op_cache.get(op_name)
    if op is None:
        # Avoid inert PartitionSpec tuples on mesh=None calls; attach replicated
        # P() specs only when distributing across a multi-device mesh.
        input_partition_specs = (P(), P(), P()) if mesh is not None else None
        op = pallas.jax_op(
            op_name,
            _kv_rope_jax,
            mesh=mesh,
            input_partition_specs=input_partition_specs,
        )
        op.register_fake(_fake_kv_rope)
        _kv_rope_op_cache[op_name] = op
    return cast(Callable[[Tensor, Tensor, Tensor], Tensor], op)


def rotate_rope(
    inputs: Tensor,
    positions: Tensor,
    rotary_emb: nn.Module,
    kv_rope_op: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    cos_sin_cache = cast(Tensor, rotary_emb.cos_sin_cache)
    max_pos = cos_sin_cache.shape[0] - 1
    # Avoid negative or out-of-range padding positions indexing cos_sin_cache OOB.
    clamped_positions = positions.clamp(0, max_pos).int()
    return kv_rope_op(inputs.contiguous(), clamped_positions, cos_sin_cache)


def active_cache(layer: DsV41CacheMixin | nn.Module) -> PagedCacheRows | Tensor:
    if isinstance(layer, DsV41CacheMixin) and layer.paged_cache_rows is not None:
        return layer.paged_cache_rows
    return cast(Tensor, layer.kv_cache)


def layer_meta(prefix: str) -> AttentionMetadata | None:
    attn_metadata = forward_context.get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return cast(AttentionMetadata | None, attn_metadata.get(prefix))
    return cast(AttentionMetadata | None, attn_metadata)


def block_table(meta: AttentionMetadata) -> Tensor:
    num_seqs = cast(Tensor, meta.seq_lens).shape[0]
    return cast(Tensor, meta.block_tables).reshape(num_seqs, -1)


def query_token_requests(query_start_loc: Tensor, positions: Tensor) -> Tensor:
    rows = torch.arange(positions.shape[0], dtype=torch.int32, device=positions.device)
    query_ends = query_start_loc[1:].to(positions.device)
    return (rows.unsqueeze(1) >= query_ends.unsqueeze(0)).sum(dim=-1, dtype=torch.int32)


def token_requests(meta: AttentionMetadata, positions: Tensor) -> Tensor:
    return query_token_requests(cast(Tensor, meta.query_start_loc), positions)


def cache_slots(
    cache: PagedCacheRows | Tensor,
    meta: AttentionMetadata,
    positions: Tensor,
    *,
    compress_ratio: int = 1,
    ring: bool = False,
) -> Tensor:
    """Resolve ``int32[T]`` physical slots, masking padded, OOB, and null blocks."""
    if cast(Tensor, meta.seq_lens).numel() == 0:
        return torch.full_like(positions, -1, dtype=torch.int32)
    table = block_table(meta)
    num_cols = table.shape[1]
    if num_cols == 0:
        return torch.full_like(positions, -1, dtype=torch.int32)
    rows_per_page = cache.shape[1]
    req_indices = token_requests(meta, positions).clamp(0, table.shape[0] - 1)
    logical_positions = positions.int() // compress_ratio
    if ring:
        block_cols = torch.zeros_like(logical_positions)
        out_of_table = positions < 0
    else:
        raw_cols = logical_positions // rows_per_page
        block_cols = raw_cols.clamp(0, num_cols - 1)
        # Avoid clamping OOB positions onto the last allocated page block.
        out_of_table = (positions < 0) | (raw_cols >= num_cols)
    flat_indices = (req_indices * num_cols + block_cols).long()
    blocks = table.reshape(-1).index_select(0, flat_indices).int()
    slots = blocks * rows_per_page + logical_positions % rows_per_page
    min_block = 1 if isinstance(cache, PagedCacheRows) and cache.null_block else 0
    token_rows = torch.arange(
        positions.shape[0], dtype=torch.int32, device=positions.device
    )
    query_limit = cast(Tensor, meta.query_start_loc)[-1].to(positions.device)
    invalid = (token_rows >= query_limit) | (blocks < min_block) | out_of_table
    return slots.masked_fill(invalid, -1)


def _context_starts(key: str, num_seqs: int, device: torch.device) -> Tensor | None:
    raw = forward_context.get_forward_context().additional_kwargs.get(key)
    if not (isinstance(raw, Tensor) and raw.numel() > 0):
        return None
    seq_indices = torch.arange(num_seqs, device=device).clamp_max(raw.shape[0] - 1)
    return raw.int()[seq_indices]


def replay_starts(
    num_seqs: int,
    device: torch.device,
    *,
    ced_decoder: bool = False,
) -> Tensor | None:
    window_starts = _context_starts("deepseek_v41_window_starts", num_seqs, device)
    decoder_starts = _context_starts("deepseek_v41_replay_starts", num_seqs, device)
    if ced_decoder:
        if window_starts is not None and decoder_starts is not None:
            return torch.maximum(window_starts, decoder_starts)
        return decoder_starts if decoder_starts is not None else window_starts
    return window_starts if window_starts is not None else decoder_starts


def not_replayed(
    meta: AttentionMetadata,
    positions: Tensor,
    window_size: int,
) -> Tensor:
    """Return ``bool[T]`` mask outside replay span ``[start, start + window_size)``."""
    num_seqs = max(cast(Tensor, meta.seq_lens).shape[0], 1)
    starts = replay_starts(num_seqs, positions.device)
    if starts is None:
        return torch.ones_like(positions, dtype=torch.bool)
    req_indices = token_requests(meta, positions).clamp_max(starts.shape[0] - 1)
    seq_starts = starts[req_indices]
    in_replay_window = (
        (seq_starts > 0)
        & (positions >= seq_starts)
        & (positions < seq_starts + window_size)
    )
    return ~in_replay_window


__all__ = [
    "active_cache",
    "block_table",
    "cache_slots",
    "get_kv_rope_op",
    "layer_meta",
    "not_replayed",
    "query_token_requests",
    "replay_starts",
    "rotate_rope",
    "token_requests",
]
