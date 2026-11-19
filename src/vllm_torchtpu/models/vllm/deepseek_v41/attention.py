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
"""DeepSeek-V4.1 sparse MLA and paged sliding-window attention layer for TPU."""

from __future__ import annotations

import copy
import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast

import jax
import torch
import torch.nn as nn
import torch.nn.functional as F
from jax.sharding import PartitionSpec as P
from torch import Tensor
from torch.tpu import pallas
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers import linear as vllm_lin
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41 import attention as upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.kernels.experimental.deepseek_v41.o_projection import (
    fused_reverse_rope_wo_a_projection,
)
from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as ops
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)
from vllm_torchtpu.utils import fresh_buffer

from .cache import (
    ROPE_WORDS,
    ROW_BYTES,
    ROW_WORDS,
    DsV41CacheMixin,
    PagedCacheRows,
    write_cache,
)
from .compressor import AttentionBackend, DeepseekCompressor
from .indexer import INDEX_HEAD_DIM, DeepseekV41Indexer, IndexerCache
from .metadata import (
    active_cache,
    block_table,
    cache_slots,
    get_kv_rope_op,
    layer_meta,
    query_token_requests,
    replay_starts,
    rotate_rope,
    token_requests,
)

PACKED_ROW_BYTES: int = 584
KV_PAGE_ALIGNMENT: int = ROW_BYTES
_ROPE_TOKENS_PER_ROW: int = ROW_WORDS // ROPE_WORDS
_EMPTY_SPARSE_TOPK: int = 128
_EMPTY_SPARSE_ROPE_ROWS: int = _EMPTY_SPARSE_TOPK // _ROPE_TOKENS_PER_ROW
_DEFAULT_SWA_BLOCK_SIZE: int = 64
_SUPPORTED_COMPRESS_RATIOS: tuple[int, ...] = (0, 1, 2)
_NUM_LN_EVENTS: int = 4
_HEADS_PER_O_GROUP: int = 8
_O_PROJ_LANE_WIDTH: int = 128
_MODEL_AXIS_NAME: str = "model"
_O_PROJ_OP_PREFIX: str = "pallas::dsv41_o_proj"
_LAYER_INDEX_RE: re.Pattern[str] = re.compile(r"\.layers\.\d+(?=\.|$)")
_o_proj_op_cache: dict[str, Any] = {}


def _o_proj_jax(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    head_dim: int,
) -> jax.Array:
    return fused_reverse_rope_wo_a_projection(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale.reshape(-1),
        head_dim=head_dim,
        inverse=True,
        quantize_activations=True,
    )


def _fake_o_proj(
    x: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    wo_a: Tensor,
    wo_a_scale: Tensor,
) -> Tensor:
    del positions, cos_sin_cache, wo_a_scale
    return torch.empty((x.shape[0], wo_a.shape[-1]), dtype=x.dtype, device=x.device)


def _replace_layer_index(prefix: str, layer_id: int) -> str:
    return _LAYER_INDEX_RE.sub(f".layers.{layer_id}", prefix, count=1)


def _pool_pages_3d(bank: Tensor, page_rows: int) -> Tensor:
    # Avoid storage_offset() during torch.compile: Dynamo rejects non-Tensor
    # storage_offset() in fullgraph traces while XLA pool buffers have offset 0.
    needs_copy = not bank.is_contiguous() or (
        not torch.compiler.is_compiling() and bank.storage_offset() != 0
    )
    if needs_copy:
        bank = bank.clone(memory_format=torch.contiguous_format)
    # Avoid unconditional .view on unit-scale 3D root pools: ExecuteCompiledModelImpl
    # inserts tt_jit_execute_as_strided copies when donated 3D pools are reshaped.
    if bank.ndim == 3 and bank.shape[1] == page_rows:
        return bank
    return bank.view(-1, page_rows, ROW_WORDS)


def _cache_has_elements(
    cache: PagedCacheRows | Tensor | tuple[Tensor, Tensor] | None,
) -> bool:
    if cache is None:
        return False
    if isinstance(cache, tuple):
        return bool(cache[0].numel()) and bool(cache[1].numel())
    return bool(cache.numel())


def _translate_unaligned_topk(
    comp_cache: PagedCacheRows,
    topk_indices: Tensor,
    comp_bt: Tensor,
    comp_cu_q: Tensor,
) -> tuple[Tensor, Tensor, int]:
    rows_per_page = comp_cache.shape[1]
    num_tokens = topk_indices.shape[0]
    num_seqs, table_width = comp_bt.shape
    token_rows = torch.arange(num_tokens, dtype=torch.int32, device=topk_indices.device)
    active_tok = token_rows < comp_cu_q[-1].to(topk_indices.device)
    req_ids = query_token_requests(comp_cu_q, token_rows).clamp(0, num_seqs - 1)
    clamped_topk = topk_indices.clamp_min(0)
    block_cols = clamped_topk // rows_per_page
    in_page = clamped_topk % rows_per_page
    bids = comp_bt[req_ids.long()].gather(
        1, block_cols.clamp_max(table_width - 1).long()
    )
    min_block = 1 if comp_cache.null_block else 0
    valid = (
        active_tok.unsqueeze(1)
        & (topk_indices >= 0)
        & (block_cols < table_width)
        & (bids >= min_block)
    )
    logical_slots = bids.clamp_min(0) * rows_per_page + in_page
    phys_rows = comp_cache.row_indices(logical_slots).masked_fill(~valid, -1)

    # Avoid page_size=rows_per_page when pages(comp_bt) is None: treating the
    # single pool as one page (total_rows % 4 == 0) keeps contiguous 3D layout.
    gather_page_size = comp_cache.raw.numel() // ROW_WORDS
    gather_bt = torch.zeros((num_seqs, 1), dtype=torch.int32, device=comp_bt.device)
    return phys_rows, gather_bt, gather_page_size


def _gather_compressed_kv(
    comp_cache: PagedCacheRows | tuple[Tensor, Tensor],
    topk_indices: Tensor,
    comp_bt: Tensor,
    comp_cu_q: Tensor,
) -> tuple[Tensor, Tensor]:
    if isinstance(comp_cache, PagedCacheRows):
        # Production path: reshape bound single-bank HBM and RoPE pools.
        assert comp_cache.rope is not None
        mapped_bt = comp_cache.pages(comp_bt)
        if mapped_bt is None:
            # Fallback for unaligned test layouts: flatten pool into one giant
            # page and translate `topk_indices` via `row_indices()`.
            gather_topk, gather_bt, gather_page_size = _translate_unaligned_topk(
                comp_cache, topk_indices, comp_bt, comp_cu_q
            )
        else:
            # Production fast path: remap only `comp_bt` to sub-page IDs and
            # keep `topk_indices` untouched.
            gather_page_size = comp_cache.shape[1]
            gather_topk = topk_indices
            gather_bt = mapped_bt
        rope_rows = gather_page_size // _ROPE_TOKENS_PER_ROW
        kv_pool = _pool_pages_3d(comp_cache.raw, gather_page_size)
        rope_pool = _pool_pages_3d(comp_cache.rope, rope_rows)
    else:
        # Unit-test fallback: tests may pass a (kv_pool, rope_pool) tuple.
        kv_pool, rope_pool = comp_cache
        gather_page_size = kv_pool.shape[1]
        gather_topk = topk_indices
        gather_bt = comp_bt

    return ops.attention_op.dsv41_gather_sparse_fp8_kv(
        kv_pool,
        rope_pool,
        gather_topk,
        gather_bt,
        comp_cu_q,
        gather_page_size,
    )


# pyrefly: ignore[inconsistent-inheritance]
class SWACache(DsV41CacheMixin, DeepseekV4SWACache):
    """Unpacked ``bfloat16`` sliding-window KV cache on TPU word-row pools."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        block_size: int = _DEFAULT_SWA_BLOCK_SIZE,
    ) -> None:
        if dtype != torch.bfloat16:
            raise ValueError(f"SWACache requires dtype=torch.bfloat16, got {dtype}")
        DeepseekV4SWACache.__init__(
            self,
            head_dim=head_dim,
            window_size=window_size,
            dtype=dtype,
            prefix=prefix,
            cache_config=cache_config,
            backend_cls=AttentionBackend,
            block_size=block_size,
            bounded_replay=cache_config.swa_bounded_replay,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        spec = cast(MLAAttentionSpec, super().get_kv_cache_spec(vllm_config))
        # Avoid inheriting upstream CUDA fp8_ds_mla 584B/576B overrides: TPU stores
        # unpacked bfloat16 SWA rows aligned to 512B word-row pools.
        return replace(
            spec,
            dtype=torch.bfloat16,
            alignment=ROW_BYTES,
            state_content_bytes=None,
            page_size_padded=None,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AttentionBackend


# pyrefly: ignore[inconsistent-inheritance]
class DeepseekV41Attention(DsV41CacheMixin, upstream.DeepseekV4Attention):
    """DeepSeek-V4.1 sparse MLA + paged sliding-window attention layer for TPU."""

    backend_cls = AttentionBackend
    swa_backend_cls = AttentionBackend
    use_fp8_ds_mla_layout = False

    @property
    def kv_rope_op(self) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        return get_kv_rope_op()

    @property
    def o_proj_op(
        self,
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
        """Return cached ``pallas::dsv41_o_proj`` op for fused inverse-RoPE + wo_a."""
        self._assert_o_proj_supported()
        head_dim = int(self.head_dim)
        mesh = get_pallas_op_mesh()
        # Avoid colliding in PyTorch's global custom-op registry across mesh modes:
        # registering a multi-device mesh under the same name would clobber mesh=None.
        op_name = (
            f"{_O_PROJ_OP_PREFIX}_d{head_dim}"
            if mesh is None
            else (
                f"{_O_PROJ_OP_PREFIX}_d{head_dim}"
                f"_mesh_{'_'.join(mesh.axis_names)}_{mesh.size}"
            )
        )
        op = _o_proj_op_cache.get(op_name)
        if op is None:
            # Avoid inert PartitionSpec tuples on mesh=None calls.
            input_partition_specs = (
                (P(None, _MODEL_AXIS_NAME, None), P(), P(), P(), P())
                if mesh is not None
                else None
            )
            op = pallas.jax_op(
                op_name,
                functools.partial(_o_proj_jax, head_dim=head_dim),
                mesh=mesh,
                input_partition_specs=input_partition_specs,
            )
            op.register_fake(_fake_o_proj)
            _o_proj_op_cache[op_name] = op
        return cast(Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor], op)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Transpose ``wo_a.weight`` to ``(8 * head_dim, G * o_lora_rank)`` once."""
        del act_dtype
        weight = cast(Tensor, self.wo_a.weight)
        transposed = (
            self.n_local_heads * self.head_dim // self.n_local_groups,
            self.n_local_groups * self.o_lora_rank,
        )
        if tuple(weight.shape) == transposed:
            return
        if tuple(weight.shape) != transposed[::-1]:
            raise ValueError(
                f"{self.prefix}.wo_a: expected weight shape "
                f"{list(transposed[::-1])}, got {list(weight.shape)}."
            )
        out = fresh_buffer(weight.data.t())
        self.wo_a.weight = nn.Parameter(out, requires_grad=False)

    def _assert_o_proj_supported(self) -> None:
        weight = cast(Tensor, self.wo_a.weight)
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{self.prefix}.wo_a: fused o-projection requires fp8_e4m3fn "
                f"weight, got {weight.dtype}."
            )
        scale = dict(self.wo_a.named_parameters(recurse=False)).get("weight_scale")
        if scale is None or scale.ndim != 1 or scale.shape[0] != weight.shape[-1]:
            raise ValueError(
                f"{self.prefix}.wo_a: fused o-projection requires 1D per-channel "
                f"weight_scale of shape [{weight.shape[-1]}], got "
                f"{None if scale is None else list(scale.shape)}."
            )
        if (
            self.n_local_heads % self.n_local_groups != 0
            or self.n_local_heads // self.n_local_groups != _HEADS_PER_O_GROUP
        ):
            raise ValueError(
                f"{self.prefix}.wo_a: fused o-projection requires "
                f"{_HEADS_PER_O_GROUP} heads per group, got "
                f"{self.n_local_heads}/{self.n_local_groups}."
            )
        if self.head_dim % _O_PROJ_LANE_WIDTH != 0:
            raise ValueError(
                f"{self.prefix}.wo_a: head_dim {self.head_dim} must be a multiple "
                f"of {_O_PROJ_LANE_WIDTH}."
            )
        rotary_dim = self.rotary_emb.cos_sin_cache.shape[-1]
        if rotary_dim % 2 != 0 or rotary_dim > _O_PROJ_LANE_WIDTH:
            raise ValueError(
                f"{self.prefix}.wo_a: rotary_dim {rotary_dim} must be even and "
                f"at most {_O_PROJ_LANE_WIDTH}."
            )

    # pyrefly: ignore[bad-override]
    def _o_proj(self, attn_out: Tensor, positions: Tensor) -> Tensor:
        if attn_out.dtype != torch.bfloat16:
            raise ValueError(
                f"{self.prefix}: fused o-projection requires bf16 activations, "
                f"got {attn_out.dtype}."
            )
        if attn_out.shape[1:] != (self.n_local_heads, self.head_dim):
            raise ValueError(
                f"{self.prefix}: fused o-projection requires shape "
                f"[T, {self.n_local_heads}, {self.head_dim}], got "
                f"{list(attn_out.shape)}."
            )
        cos_sin_cache = cast(Tensor, self.rotary_emb.cos_sin_cache)
        max_pos = cos_sin_cache.shape[0] - 1
        # Avoid negative or out-of-range padding positions indexing cos_sin_cache OOB.
        clamped_positions = positions.clamp(0, max_pos).int()
        z = self.o_proj_op(
            attn_out,
            clamped_positions,
            cos_sin_cache,
            cast(Tensor, self.wo_a.weight),
            cast(Tensor, self.wo_a.weight_scale),
        )
        out = self.wo_b(z)
        return out[0] if isinstance(out, tuple) else out

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    # pyrefly: ignore[bad-override]
    def _indexer_cls(self, k_cache: IndexerCache | None) -> type[DeepseekV41Indexer]:
        del k_cache
        return DeepseekV41Indexer

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: Tensor | None = None,
        aux_stream_list: Sequence[object] | None = None,
        candidate_block_buffer: Tensor | None = None,
    ) -> None:
        # Avoid upstream.DeepseekV4Attention.__init__: hardcodes torch.cuda.Event,
        # Triton warmups, and CUDA-specific DeepseekV4SWACache / DeepseekV4IndexerCache.
        nn.Module.__init__(self)
        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        if cache_config is None:
            raise ValueError("DeepseekV41Attention requires vllm_config.cache_config")

        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)
        self.layer_id = layer_id
        self.prefix = prefix
        self.window_size: int = hf_config.sliding_window
        self.hidden_size: int = hf_config.hidden_size
        self.n_heads: int = hf_config.num_attention_heads
        self.n_local_heads: int = self.n_heads // tp_size
        self.q_lora_rank: int = hf_config.q_lora_rank
        self.o_lora_rank: int = hf_config.o_lora_rank
        self.head_dim: int = hf_config.head_dim
        self.rope_head_dim: int = hf_config.qk_rope_head_dim
        self.nope_head_dim: int = self.head_dim - self.rope_head_dim
        self.n_groups: int = hf_config.o_groups
        self.n_local_groups: int = self.n_groups // tp_size

        ratios = hf_config.compress_ratios
        compress_ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(f"Unsupported compress_ratio={compress_ratio}")
        self.compress_ratio = compress_ratio

        self.kv_source_layers = tuple(hf_config.kv_source_layer_ids)
        self.index_source_layers = tuple(hf_config.index_source_layer_ids)
        self.candidate_source_layer: int = hf_config.candidate_source_layer_id
        self.candidate_block_size: int = hf_config.candidate_block_size
        self.candidate_topk_blocks: int = hf_config.candidate_topk_blocks

        is_backbone_layer = layer_id < hf_config.num_hidden_layers
        self.is_kv_source = is_backbone_layer and layer_id in self.kv_source_layers
        self.is_index_source = (
            is_backbone_layer and layer_id in self.index_source_layers
        )
        if compress_ratio > 0:
            if not self.kv_source_layers or not self.index_source_layers:
                raise ValueError(
                    "DeepSeek V4.1 requires kv_source_layer_ids / "
                    "index_source_layer_ids in the config for compressed "
                    f"layers (layer {layer_id} has "
                    f"compress_ratio={compress_ratio})."
                )
            prior_kv_sources = [
                src_layer
                for src_layer in self.kv_source_layers
                if src_layer <= layer_id
            ]
            if not prior_kv_sources:
                raise ValueError(
                    f"No kv_source_layer_id <= {layer_id} in {self.kv_source_layers}."
                )
            self.kv_source_layer_id = max(prior_kv_sources)
            prior_idx_sources = [
                src_layer
                for src_layer in self.index_source_layers
                if src_layer <= layer_id
            ]
            if not prior_idx_sources:
                raise ValueError(
                    f"No index_source_layer_id <= {layer_id} in "
                    f"{self.index_source_layers}."
                )
            self.index_source_layer_id = max(prior_idx_sources)
        else:
            self.kv_source_layer_id = None
            self.index_source_layer_id = None

        self.eps: float = hf_config.rms_norm_eps
        self.scale: float = self.head_dim**-0.5
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        sink_init = torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32)
        self.attn_sink = nn.Parameter(sink_init, requires_grad=False)

        self.fused_wqa_wkv = vllm_lin.MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        q_out_dim = self.n_heads * self.head_dim
        o_mid_dim = self.n_groups * self.o_lora_rank
        self.wq_b = vllm_lin.ColumnParallelLinear(
            self.q_lora_rank,
            q_out_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = vllm_lin.ColumnParallelLinear(
            q_out_dim // self.n_groups,
            o_mid_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        # pyrefly: ignore[bad-argument-type]
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = vllm_lin.RowParallelLinear(
            o_mid_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        rope = upstream.build_deepseek_v4_rope(
            hf_config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=hf_config.max_position_embeddings,
            compress_ratio=compress_ratio,
        )
        max_model_len = vllm_config.model_config.max_model_len
        if rope.cos_sin_cache.shape[0] > max_model_len:
            # Avoid mutating vLLM's memoized _ROPE_DICT singleton _buffers in
            # place: truncating the shared rope breaks later callers with larger
            # max_model_len.
            rope = copy.deepcopy(rope)
            rope.cos_sin_cache = rope.cos_sin_cache[:max_model_len].clone()
        self.rotary_emb = rope
        self.indexer_rotary_emb = rope

        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        static_ctx = vllm_config.compilation_config.static_forward_context
        if prefix in static_ctx:
            raise ValueError(f"Duplicate layer name: {prefix}")
        self._static_forward_context = static_ctx
        static_ctx[prefix] = self

        self.kv_cache = torch.tensor([])
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.indexer: DeepseekV41Indexer | None = None
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.compressor: DeepseekCompressor | None = None

        source_prefix = (
            None
            if self.kv_source_layer_id is None
            else _replace_layer_index(prefix, self.kv_source_layer_id)
        )
        if self.is_index_source:
            owner_prefix = prefix if self.is_kv_source else source_prefix
            indexer_k_prefix = f"{owner_prefix}.indexer.k_cache"
            if self.is_kv_source:
                indexer_k_cache = IndexerCache(
                    head_dim=INDEX_HEAD_DIM,
                    dtype=torch.bfloat16,
                    prefix=indexer_k_prefix,
                    cache_config=cache_config,
                    compress_ratio=compress_ratio,
                )
            else:
                shared_k_cache = static_ctx.get(indexer_k_prefix)
                if shared_k_cache is None:
                    raise NotImplementedError(
                        f"Indexer K cache source {indexer_k_prefix} not found on "
                        "this rank; PP splits inside a v4.1 kv-sharing group "
                        "are not supported."
                    )
                indexer_k_cache = cast(IndexerCache, shared_k_cache)
            is_candidate_source = layer_id == self.candidate_source_layer
            uses_candidates = 0 <= self.candidate_source_layer < layer_id
            active_candidate_buffer = (
                candidate_block_buffer
                if (is_candidate_source or uses_candidates)
                else None
            )
            indexer_cls = self._indexer_cls(indexer_k_cache)
            self.indexer = indexer_cls(
                vllm_config=vllm_config,
                config=hf_config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=indexer_k_cache,
                main_head_dim=self.head_dim,
                candidate_block_buffer=active_candidate_buffer,
                candidate_block_size=self.candidate_block_size,
                candidate_write=is_candidate_source,
            )

        # pyrefly: ignore[bad-assignment]
        self._prepare_and_attn_fn = self._prepare_and_attn
        self.aux_stream_list = cast(list[torch.cuda.Stream] | None, aux_stream_list)
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.ln_events: list[object | None] = [None] * _NUM_LN_EVENTS
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = max_model_len
        self.gemm_rs = None
        self.shard_fused_wqa_wkv = False
        self.kv_mxfp8 = False
        self.kv_cache_dtype = "fp8"
        self.kv_cache_torch_dtype = torch.float8_e4m3fn
        self.swa_bytes_per_token = PACKED_ROW_BYTES
        self.compressed_bytes_per_token = PACKED_ROW_BYTES
        self.kv_page_alignment = KV_PAGE_ALIGNMENT
        swa_block_size = (
            _DEFAULT_SWA_BLOCK_SIZE
            if cache_config.block_size is None
            else cache_config.block_size
        )
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.swa_cache_layer: SWACache = SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            block_size=swa_block_size,
        )
        if self.is_kv_source:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=prefix,
            )
        if compress_ratio > 0:
            assert source_prefix is not None
            if not self.is_kv_source and source_prefix not in static_ctx:
                raise NotImplementedError(
                    f"Compressed-KV source {source_prefix} not found on this "
                    "rank; PP splits inside a v4.1 kv-sharing group are not "
                    "supported."
                )
            self.compressed_cache_prefix = source_prefix
        else:
            self.compressed_cache_prefix = None
        self.dspark_block_size = 0
        self.ced_global_from_encoder = False
        self.ced_decoder = False
        self.non_causal_multi_token_decode = False

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        mla_spec = cast(MLAAttentionSpec, spec)
        # Report 512B NoPE row + 128B companion RoPE = 640B per state so vLLM's
        # `page_size_bytes` budget covers both physical banks.
        state_bytes = ROW_BYTES + self.rope_head_dim * torch.bfloat16.itemsize
        return replace(
            mla_spec,
            dtype=torch.uint8,
            alignment=ROW_BYTES,
            state_content_bytes=state_bytes,
            page_size_padded=None,
        )

    def _run_parallel_input_projections(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = (
            self.compressor.project(hidden_states)
            if self.compressor is not None
            else None
        )
        indexer_weights = (
            self.indexer.weights_proj(hidden_states)[0]
            if self.indexer is not None
            else None
        )
        return qr_kv, kv_score, indexer_weights

    def _split_qkv_and_norm(self, qr_kv: Tensor) -> tuple[Tensor, None, Tensor]:
        qr_proj, kv_latent = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return self.q_norm(qr_proj), None, self.kv_norm(kv_latent)

    def project_context_kv(self, hidden_states: Tensor) -> Tensor:
        _, _, kv_latent = self._split_qkv_and_norm(
            self._fused_wqa_wkv_gemm(hidden_states)
        )
        return kv_latent

    def rotate(self, inputs: Tensor, positions: Tensor) -> Tensor:
        return rotate_rope(inputs, positions, self.rotary_emb, self.kv_rope_op)

    def _swa_state(self) -> tuple[PagedCacheRows | Tensor, AttentionMetadata | None]:
        swa_layer = self.swa_cache_layer
        return active_cache(swa_layer), layer_meta(swa_layer.prefix)

    def _write_swa_kv(
        self,
        kv_rotated: Tensor,
        positions: Tensor,
        after_read: bool = False,
    ) -> None:
        swa_cache, swa_meta = self._swa_state()
        if (
            swa_meta is None
            or not _cache_has_elements(swa_cache)
            or not cast(Tensor, swa_meta.seq_lens).numel()
        ):
            return
        slots = cache_slots(swa_cache, swa_meta, positions)
        query_ends = cast(Tensor, swa_meta.query_start_loc)[1:].to(positions.device)
        req_indices = token_requests(swa_meta, positions).clamp_max(
            query_ends.shape[0] - 1
        )
        table_capacity = block_table(swa_meta).shape[1] * swa_cache.shape[1]
        window_cap = min(self.window_size + self.dspark_block_size, table_capacity)
        token_rows = torch.arange(positions.shape[0], device=positions.device)
        keep_mask = (slots >= 0) & (token_rows >= query_ends[req_indices] - window_cap)
        write_cache(swa_cache, slots, kv_rotated, keep_mask, after_read=after_read)

    def _gather_seq_swa(self, live_kv: Tensor | None = None) -> Tensor | None:
        swa_cache, swa_meta = self._swa_state()
        # Output shape & role (S=num_seqs, W=window_size, B=dspark_block_size):
        # - Causal (non_causal=False): [S, W, D] with committed swa_cache rows
        #   only; current-step KV is appended via torch.cat in forward_mqa.
        # - DSpark (non_causal=True): [S, W + B, D] with committed swa_cache
        #   rows + current-step live_kv together (no swa_cache write or cat).
        non_causal = self.non_causal_multi_token_decode
        block_tokens = max(0, self.dspark_block_size) if non_causal else 0
        eff_window = self.window_size + block_tokens
        if (
            swa_meta is None
            or not _cache_has_elements(swa_cache)
            or not cast(Tensor, swa_meta.seq_lens).numel()
            or eff_window <= 0
        ):
            return None

        swa_table = block_table(swa_meta).to(torch.int64)
        table_width = swa_table.shape[1]
        if table_width == 0:
            return None

        seq_lens = cast(Tensor, swa_meta.seq_lens).to(torch.int64)
        cu_query = cast(Tensor, swa_meta.query_start_loc).to(torch.int64)
        query_lens = cu_query[1:] - cu_query[:-1]
        live_count = 0 if live_kv is None else live_kv.shape[0]
        # Only used on DSpark draft layers L40..L42 where `live_per_seq` equals
        # `query_lens`; `minimum` and `clamp` are defensive bounds checks.
        live_per_seq = torch.minimum(
            query_lens, (live_count - cu_query[:-1]).clamp_min(0)
        ).clamp(0, block_tokens)

        # Inclusive right edge `window_end` of the gathered window; e.g., for
        # `seq_lens` = 25 with 20 cached tokens 0..19 and 5 new tokens 20..24:
        if non_causal:
            # DSpark splices 20..24 from `live_kv` right here, so set
            # `window_end` to 24.
            window_end = seq_lens - 1
            committed_end: Tensor | None = seq_lens - live_per_seq
            window_base = (window_end - eff_window + 1).clamp_min(0)
        else:
            # Causal appends 20..24 later in `forward_mqa`, not here, so set
            # `window_end` to 19.
            window_end = seq_lens - query_lens - 1
            committed_end = None
            window_base = window_end - eff_window + 1

        window_offsets = torch.arange(
            eff_window, dtype=torch.int64, device=seq_lens.device
        )
        kv_positions = window_base[:, None] + window_offsets
        lower_bounds = replay_starts(
            seq_lens.shape[0], seq_lens.device, ced_decoder=self.ced_decoder
        )
        min_pos = (
            lower_bounds.to(torch.int64).clamp_min(0).unsqueeze(1)
            if lower_bounds is not None
            else 0
        )

        clamped_pos = kv_positions.clamp_min(0)
        rows_per_page = swa_cache.shape[1]
        raw_cols = clamped_pos // rows_per_page
        page_cols = raw_cols.clamp_max(table_width - 1)
        blocks = swa_table.gather(1, page_cols)
        min_block = (
            1 if isinstance(swa_cache, PagedCacheRows) and swa_cache.null_block else 0
        )
        valid_committed = (
            (kv_positions >= min_pos) & (raw_cols < table_width) & (blocks >= min_block)
        )
        if committed_end is not None:
            valid_committed = valid_committed & (
                kv_positions < committed_end.unsqueeze(1)
            )
        if isinstance(swa_cache, PagedCacheRows) and swa_cache.supports_paged_windows:
            # Avoid per-token row gathers: paged bursts prevent 1-row DMAs on TPU7x.
            gathered_rows = swa_cache.gather_paged_windows(
                swa_table, window_base, eff_window
            )
        else:
            gathered_rows = swa_cache[blocks.clamp_min(0), clamped_pos % rows_per_page]
        # Avoid 0.0 * x: select zeroes unmapped/null slots without NaN propagation.
        swa_out = torch.where(
            valid_committed.unsqueeze(-1), gathered_rows.bfloat16(), 0
        )

        if committed_end is None or live_kv is None or block_tokens <= 0:
            return swa_out

        # Avoid branching on symbolic live_count > 0: trailing sentinel row keeps
        # zero-token dummy forwards in-bounds with zero Dynamo shape guards.
        padded_live = F.pad(live_kv.bfloat16(), (0, 0, 0, 1))
        live_offset = kv_positions - committed_end.unsqueeze(1)
        in_live_block = (live_offset >= 0) & (live_offset < live_per_seq.unsqueeze(1))
        live_row_idx = (
            cu_query[:-1].unsqueeze(1) + live_offset.clamp(0, block_tokens - 1)
        ).clamp(0, live_count)
        return torch.where(
            in_live_block.unsqueeze(-1), padded_live[live_row_idx], swa_out
        )

    def _compress_kv(
        self,
        kv_score: Tensor,
        positions: Tensor,
        produce_indexer_k: bool = False,
    ) -> Tensor:
        assert self.compressor is not None
        latent = self.compressor(kv_score, positions)
        self.compressor.insert_cache(latent, positions, self.rotary_emb)
        if produce_indexer_k and self.indexer is not None:
            self.indexer.produce_k(latent, positions, self.indexer_rotary_emb)
        return latent

    def write_encoder_global_kv(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        *,
        sp_sharded: bool = False,
    ) -> None:
        if self.compressor is None:
            return
        kv_score = self.compressor.project(hidden_states)
        if sp_sharded:
            kv_score = sp_all_gather(kv_score)[: positions.shape[0]]
        # Publish indexer K eagerly here because this KV-only encoder hook skips
        # self.indexer(...), whereas _prepare_and_attn leaves produce_indexer_k
        # False so self.indexer.forward() writes K once without duplicating it.
        self._compress_kv(kv_score, positions, produce_indexer_k=True)

    def forward_kv(self, positions: Tensor, hidden_states: Tensor) -> None:
        qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)
        _, _, kv_latent = self._split_qkv_and_norm(qr_kv)
        self._write_swa_kv(self.rotate(kv_latent, positions), positions)
        if self.compressor is not None and kv_score is not None:
            self._compress_kv(kv_score, positions, produce_indexer_k=True)

    # pyrefly: ignore[bad-override-param-name]
    def _prepare_and_attn(
        self,
        hidden_states: Tensor,
        qr_proj: Tensor,
        kv_latent: Tensor,
        qr_scale: Tensor | None,
        kv_score: Tensor | None,
        indexer_weights: Tensor | None,
        positions: Tensor,
        attn_output: Tensor,
    ) -> None:
        q_heads = self._wq_b_proj(qr_proj, qr_scale).view(
            -1, self.n_local_heads, self.head_dim
        )
        q_rotated = self.rotate(q_heads, positions)
        kv_rotated = self.rotate(kv_latent, positions)

        if self.non_causal_multi_token_decode:
            pre_write_swa: Tensor | None = None
            compressed_latent: Tensor | None = None
        else:
            pre_write_swa = self._gather_seq_swa()
            # Pass after_read so XLA orders the prefix gather before circular
            # page scatters that may reuse an outgoing SWA block in this step.
            self._write_swa_kv(
                kv_rotated, positions, after_read=pre_write_swa is not None
            )
            # On the CED boundary layer (e.g., L20), write_encoder_global_kv
            # already populated global KV and k_cache at encoder end; skip
            # re-compressing (and produce_k via None) in the decoder stage.
            if (
                self.compressor is not None
                and not self.ced_global_from_encoder
                and kv_score is not None
            ):
                compressed_latent = self._compress_kv(kv_score, positions)
            else:
                compressed_latent = None

        if self.indexer is not None and indexer_weights is not None:
            idx_q, idx_q_scale, idx_weights = self.indexer(
                qr_proj,
                compressed_latent,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
                qr_scale,
            )
            assert idx_q is not None and idx_weights is not None
            self.indexer.indexer_op(
                hidden_states, (idx_q, idx_q_scale), positions, idx_weights
            )

        self.forward_mqa(q_rotated, kv_rotated, positions, attn_output, pre_write_swa)

    def forward_mqa(
        self,
        q: Tensor,
        kv: Tensor,
        positions: Tensor,
        output: Tensor,
        pre_write_swa: Tensor | None = None,
    ) -> None:
        num_tokens = q.shape[0]
        device = q.device
        swa_cache, swa_meta = self._swa_state()
        if (
            swa_meta is None
            or not _cache_has_elements(swa_cache)
            or not cast(Tensor, swa_meta.seq_lens).numel()
        ):
            output.zero_()
            return

        seq_lens = cast(Tensor, swa_meta.seq_lens).int()
        cu_q_lens = cast(Tensor, swa_meta.query_start_loc).int()
        seq_swa = self._gather_seq_swa(kv) if pre_write_swa is None else pre_write_swa
        lower_bounds = replay_starts(
            seq_lens.shape[0], device, ced_decoder=self.ced_decoder
        )

        if self.non_causal_multi_token_decode:
            # Reuse the causal kernel (kv_pos <= pos_t) for bidirectional DSpark
            # blocks by viewing the batch as T 1-token decode pseudo-sequences
            # all ending at their parent request's seq_len - 1.
            req_indices = (
                token_requests(swa_meta, positions)
                .clamp(0, seq_lens.shape[0] - 1)
                .long()
            )
            # Clamp empty 0-length slots to 1 so kernel's (seq_len - 1) >= 0.
            seq_lens = (
                (seq_lens.to(torch.int64)[req_indices] - 1).clamp_min(0).add(1).int()
            )
            cu_q_lens = torch.arange(
                num_tokens + 1, dtype=torch.int32, device=device
            ).clamp_max(cu_q_lens[-1])
            eff_window = self.window_size + self.dspark_block_size
            swa_kv = None if seq_swa is None else seq_swa[req_indices]
            swa_min_pos = None if lower_bounds is None else lower_bounds[req_indices]
        else:
            eff_window = self.window_size
            flat_swa = None if seq_swa is None else seq_swa.reshape(-1, self.head_dim)
            swa_kv = (
                None if flat_swa is None else torch.cat((flat_swa, kv.bfloat16()), 0)
            )
            swa_min_pos = lower_bounds

        comp_prefix = self.compressed_cache_prefix
        comp_layer = (
            cast(
                DsV41CacheMixin | None,
                self._static_forward_context.get(comp_prefix),
            )
            if comp_prefix is not None
            else None
        )
        comp_cache = active_cache(comp_layer) if comp_layer is not None else None
        comp_meta = layer_meta(comp_prefix) if comp_prefix is not None else None
        comp_bt = None if comp_meta is None else block_table(comp_meta).int()
        if (
            self.compress_ratio > 0
            and _cache_has_elements(comp_cache)
            and isinstance(comp_cache, (PagedCacheRows, tuple))
            and comp_meta is not None
            and comp_bt is not None
            and comp_bt.numel() > 0
            and self.topk_indices_buffer is not None
        ):
            # Global MLA layers (e.g., L2..L39): gather top-K compressed NoPE
            # and RoPE keys from the shared source pool via SparseCore.
            topk_indices = self.topk_indices_buffer[:num_tokens]
            gathered_nope, gathered_rope = _gather_compressed_kv(
                comp_cache,
                topk_indices,
                comp_bt,
                cast(Tensor, comp_meta.query_start_loc).int(),
            )
            kv_ratio = self.compress_ratio
        else:
            # Pure SWA layers (e.g., L0..L1, DSpark L40..L42): pass -1 top-K
            # placeholders so the fused kernel skips sparse MLA and runs SWA.
            topk_indices = seq_lens.new_full((num_tokens, _EMPTY_SPARSE_TOPK), -1)
            gathered_nope = topk_indices.new_zeros(
                (num_tokens, _EMPTY_SPARSE_TOPK, ROW_WORDS)
            )
            gathered_rope = topk_indices.new_zeros(
                (num_tokens, _EMPTY_SPARSE_ROPE_ROWS, ROW_WORDS)
            )
            kv_ratio = max(1, self.compress_ratio)

        attn_res = ops.attention_op.dsv41_sparse_mla_swa_attention(
            q.bfloat16(),
            gathered_nope,
            gathered_rope,
            topk_indices,
            self.attn_sink.float(),
            seq_lens,
            cu_q_lens,
            self.scale,
            eff_window,
            swa_kv,
            kv_ratio,
            swa_min_pos,
        )
        output.copy_(attn_res.to(output.dtype))


__all__ = [
    "KV_PAGE_ALIGNMENT",
    "PACKED_ROW_BYTES",
    "DeepseekV41Attention",
    "SWACache",
]
