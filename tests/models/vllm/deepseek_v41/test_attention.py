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
"""Unit and TPU7x parity tests for DeepSeek-V4.1 attention and SWA layers."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
from jax.sharding import AbstractMesh, Mesh
from jax.sharding import PartitionSpec as P
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.tpu import pallas as tpu_pallas
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.ir.ops import rms_norm
from vllm.models.deepseek_v41.attention import build_deepseek_v4_rope
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.v1.core import kv_cache_planning as kcu
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    ROPE_WORDS,
    ROW_BYTES,
    ROW_WORDS,
    DsV41CacheMixin,
    PagedCacheRows,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import attention as attn_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import cache as cache_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import indexer as indexer_mod
from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
    KV_PAGE_ALIGNMENT,
    PACKED_ROW_BYTES,
    DeepseekV41Attention,
    SWACache,
)
from vllm_torchtpu.models.vllm.deepseek_v41.compressor import (
    AttentionBackend,
    DeepseekCompressor,
)
from vllm_torchtpu.models.vllm.deepseek_v41.indexer import DeepseekV41Indexer
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner import kv_cache_manager as kcm_mod
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

GIB_BYTES: int = 1 << 30
MAIN_HEAD_DIM: int = 512
ROPE_HEAD_DIM: int = 64
NOPE_HEAD_DIM: int = MAIN_HEAD_DIM - ROPE_HEAD_DIM
HIDDEN_SIZE: int = 64
Q_LORA_RANK: int = 32
O_LORA_RANK: int = 128
NUM_HEADS: int = 8
O_GROUPS: int = 1
SLIDING_WINDOW: int = 8
INDEX_HEAD_DIM: int = 128
KEYS_PER_WORD_ROW: int = 2
BF16_SHIFT_BITS: int = 16
BF16_ODD_MASK: int = -65536
FP8_E4M3_MAX: float = float(torch.finfo(torch.float8_e4m3fn).max)


@pytest.fixture
def cpu_dist_ctx(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")


class _JaxBackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem: jax.Array, env: object = None) -> _JaxBackedTensor:
        del env
        dtype_map = {
            jnp.dtype(jnp.int32): torch.int32,
            jnp.dtype(jnp.bfloat16): torch.bfloat16,
            jnp.dtype(jnp.float8_e4m3fn): torch.float8_e4m3fn,
            jnp.dtype(jnp.uint8): torch.uint8,
            jnp.dtype(jnp.float32): torch.float32,
        }
        shape = tuple(int(dim) for dim in elem.shape)
        base = torch.empty(shape, dtype=dtype_map[jnp.dtype(elem.dtype)])
        return base.as_subclass(cls)

    def __init__(self, elem: jax.Array, env: object = None) -> None:
        self._elem = elem
        self._env = env

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._elem)


class _StubModelConfig:
    def __init__(
        self,
        hf_config: DeepseekV41Config,
        max_model_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.hf_config = hf_config
        self.hf_text_config = hf_config
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.model = "deepseek-v41"
        self.tokenizer = "deepseek-v41"
        self.is_hybrid = False
        self.encoder_config = None
        self.disable_sliding_window = False
        self.quantization = None

    def get_sliding_window(self) -> int | None:
        return int(self.hf_config.sliding_window)


def _make_dsv41_vllm_config(
    *,
    block_size: int | None = 16,
    cache_dtype: str = "auto",
    model_dtype: torch.dtype = torch.bfloat16,
    max_model_len: int = 256,
    max_position_embeddings: int = 512,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 64,
    num_attention_heads: int = NUM_HEADS,
    o_groups: int = O_GROUPS,
    o_lora_rank: int = O_LORA_RANK,
    index_n_heads: int = 4,
    index_topk: int = 8,
    sliding_window: int = SLIDING_WINDOW,
    compress_ratios: list[int] | None = None,
    kv_source_layer_ids: list[int] | None = None,
    index_source_layer_ids: list[int] | None = None,
    candidate_source_layer_id: int = 19,
    candidate_block_size: int = 4,
    candidate_topk_blocks: int = 4,
    tp_size: int = 1,
    device: str = "cpu",
    swa_bounded_replay: bool = False,
) -> VllmConfig:
    hf_cfg = DeepseekV41Config(
        vocab_size=256,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=40,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        kv_lora_rank=NOPE_HEAD_DIM,
        q_lora_rank=Q_LORA_RANK,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        qk_nope_head_dim=64,
        qk_rope_head_dim=ROPE_HEAD_DIM,
        v_head_dim=64,
        head_dim=MAIN_HEAD_DIM,
        index_n_heads=index_n_heads,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=index_topk,
        sliding_window=sliding_window,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        candidate_source_layer_id=candidate_source_layer_id,
        candidate_block_size=candidate_block_size,
        candidate_topk_blocks=candidate_topk_blocks,
        text_config={"rope_scaling": {"rope_type": "default"}},
    )
    hf_cfg.compress_ratios = (
        [0, 0] + [2] * 18 + [1] * 20
        if compress_ratios is None
        else list(compress_ratios)
    )
    hf_cfg.kv_source_layer_ids = (
        [2, 8, 14, 20] if kv_source_layer_ids is None else list(kv_source_layer_ids)
    )
    hf_cfg.index_source_layer_ids = (
        [2, 8, 14, 19, 20]
        if index_source_layer_ids is None
        else list(index_source_layer_ids)
    )

    cache_cfg = CacheConfig(
        block_size=cast(Any, block_size),
        cache_dtype=cast(Any, cache_dtype),
        swa_bounded_replay=swa_bounded_replay,
    )
    cache_cfg.kv_cache_layout = "BLHNC"
    vcfg = VllmConfig(
        cache_config=cache_cfg,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device=cast(Any, device)),
        attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=5),
    )
    vcfg.parallel_config.tensor_parallel_size = tp_size
    vcfg.model_config = cast(Any, _StubModelConfig(hf_cfg, max_model_len, model_dtype))
    vcfg.scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )
    return vcfg


def _make_attn_meta(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> AttentionMetadata:
    dev = seq_lens.device
    pos = (
        torch.zeros(1, dtype=torch.int32, device=dev)
        if positions is None
        else positions
    )
    return AttentionMetadata(
        input_positions=pos,
        block_tables=block_tables.reshape(-1),
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=torch.zeros(3, dtype=torch.int32, device=dev),
    )


def _forward_ctx(
    attn_metadata: Any,
    additional_kwargs: dict[str, Any] | None = None,
) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping={},
        additional_kwargs={} if additional_kwargs is None else dict(additional_kwargs),
    )


def _init_attention_weights(
    layer: DeepseekV41Attention, seed: int, *, fp8_wo_a: bool = False
) -> None:
    cos_sin_f32 = {
        name: buf.detach().clone().to(torch.float32)
        for name, buf in layer.named_buffers()
        if name.endswith("cos_sin_cache")
    }
    layer.to(torch.bfloat16)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, buf in layer.named_buffers():
            if name in cos_sin_f32:
                buf.data = cos_sin_f32[name]
        for name, param in layer.named_parameters():
            if name == "attn_sink":
                param.data = torch.linspace(
                    -1.5, 0.5, param.numel(), dtype=torch.float32, device=param.device
                )
            elif "norm" in name or "weight" not in name:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
        if fp8_wo_a:
            w_f32 = cast(torch.Tensor, layer.wo_a.weight).float()
            scale = w_f32.abs().amax(dim=1).clamp_min(1e-4) / 240.0
            w_fp8 = (w_f32 / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            layer.wo_a.weight = nn.Parameter(w_fp8, requires_grad=False)
            layer.wo_a.weight_scale = nn.Parameter(
                scale.to(torch.float32), requires_grad=False
            )


def _cpu_rope(
    inputs: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    rot_dim = cos_sin_cache.shape[-1]
    cs = cos_sin_cache[positions.long()].float()
    if inputs.ndim == 3:
        cs = cs.unsqueeze(1)
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    tail = inputs[..., -rot_dim:].float()
    rotated = torch.stack((-tail[..., 1::2], tail[..., 0::2]), dim=-1).flatten(-2)
    out_tail = (tail * cos + rotated * sin).to(inputs.dtype)
    return torch.cat((inputs[..., :-rot_dim], out_tail), dim=-1)


def _cpu_inverse_rope_wo_a(
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
) -> torch.Tensor:
    rot_dim = cos_sin_cache.shape[-1]
    cs = cos_sin_cache[positions.clamp(0, cos_sin_cache.shape[0] - 1).long()].float()
    cos, sin = cs.unsqueeze(1).chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    tail = attn_out[..., -rot_dim:].float()
    rot_inv = torch.stack((tail[..., 1::2], -tail[..., 0::2]), dim=-1).flatten(-2)
    unrotated = torch.cat(
        (attn_out[..., :-rot_dim].float(), tail * cos + rot_inv * sin), dim=-1
    )
    num_tokens, num_heads, head_dim = unrotated.shape
    num_groups = num_heads // 8
    reduction = 8 * head_dim
    lora_rank = wo_a.shape[1] // num_groups
    x_g = unrotated.reshape(num_tokens, num_groups, reduction)
    w_g = wo_a.float().reshape(reduction, num_groups, lora_rank)
    s_g = wo_a_scale.float().reshape(1, num_groups, lora_rank)
    out = torch.einsum("tgd,dgr->tgr", x_g, w_g) * s_g
    return out.reshape(num_tokens, num_groups * lora_rank).to(attn_out.dtype)


def _ref_pack_fp8_kv_rows(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent_f32 = latents.float()
    subtile3_mask = torch.arange(ROW_WORDS, device=latents.device) < 64
    s0 = latent_f32[:, :ROW_WORDS]
    s1 = latent_f32[:, ROW_WORDS : 2 * ROW_WORDS]
    s2 = latent_f32[:, 2 * ROW_WORDS : 3 * ROW_WORDS]
    s3 = latent_f32[:, 3 * ROW_WORDS : MAIN_HEAD_DIM]
    amax = torch.maximum(
        torch.maximum(s0.abs(), s1.abs()),
        torch.maximum(s2.abs(), torch.where(subtile3_mask[None, :], s3.abs(), 0.0)),
    )
    amax = torch.maximum(amax, torch.roll(amax, shifts=64, dims=1)).clamp_min(1e-12)
    exp_i32 = torch.ceil(torch.log2(amax / FP8_E4M3_MAX)).to(torch.int32)
    inv_scale = ((127 - exp_i32) << 23).view(torch.float32)

    def _q_byte(sub: torch.Tensor) -> torch.Tensor:
        q_u8 = (sub * inv_scale).to(torch.float8_e4m3fn).view(torch.uint8)
        return q_u8.to(torch.int64)

    top_byte = torch.where(
        subtile3_mask[None, :],
        _q_byte(s3),
        (exp_i32 + 127).clamp(0, 255).to(torch.int64),
    )
    nope_i64 = _q_byte(s0) | (_q_byte(s1) << 8) | (_q_byte(s2) << 16) | (top_byte << 24)
    nope_i32 = nope_i64.to(torch.uint32).view(torch.int32)

    tail_u16 = (
        latents.bfloat16()[:, 3 * ROW_WORDS : MAIN_HEAD_DIM]
        .view(torch.uint16)
        .to(torch.int64)
    )
    rope_low_u16 = torch.roll(tail_u16, shifts=64, dims=1)[:, :ROPE_WORDS]
    rope_high_u16 = torch.roll(tail_u16, shifts=32, dims=1)[:, :ROPE_WORDS]
    rope_i32 = (rope_low_u16 | (rope_high_u16 << 16)).to(torch.uint32).view(torch.int32)
    return nope_i32, rope_i32


def _dequantize_packed_fp8_row(
    nope_words: torch.Tensor, rope_quarter: torch.Tensor
) -> torch.Tensor:
    u32 = nope_words.numpy().astype(np.uint32)
    top_byte = (u32 >> 24) & 0xFF
    rolled = np.roll(top_byte, 64, axis=-1)
    lane_ids = np.arange(ROW_WORDS, dtype=np.int32)
    scale_u8 = np.where(lane_ids < 64, rolled, top_byte).astype(np.uint8)
    scale_f32 = np.exp2(scale_u8.astype(np.float32) - 127.0)
    slabs: list[np.ndarray] = []
    for slab in range(3):
        b = ((u32 >> (8 * slab)) & 0xFF).astype(np.uint8)
        fp8 = torch.from_numpy(b.copy()).view(torch.float8_e4m3fn).float().numpy()
        slabs.append(fp8 * scale_f32)
    tail_b = top_byte[..., :64].astype(np.uint8)
    tail_fp8 = torch.from_numpy(tail_b.copy()).view(torch.float8_e4m3fn).float().numpy()
    slabs.append(tail_fp8 * scale_f32[..., :64])
    r_u32 = rope_quarter.numpy().astype(np.uint32)
    r_lo = torch.from_numpy(((r_u32 & 0xFFFF) << 16).astype(np.int32)).view(
        torch.float32
    )
    r_hi = torch.from_numpy((r_u32 & 0xFFFF0000).astype(np.int32)).view(torch.float32)
    rope_f32 = torch.cat((r_lo, r_hi), dim=-1).numpy()
    slabs.append(rope_f32)
    return torch.from_numpy(np.concatenate(slabs, axis=-1)).float()


def _unpack_split_bf16_keys(words: torch.Tensor) -> torch.Tensor:
    keys_words = words.reshape(-1, INDEX_HEAD_DIM // 2)
    even = (keys_words << BF16_SHIFT_BITS).view(torch.float32).to(torch.bfloat16)
    odd = (keys_words & BF16_ODD_MASK).view(torch.float32).to(torch.bfloat16)
    return torch.cat((even, odd), dim=-1).float()


def _ref_paged_indexer_topk(
    query: torch.Tensor,
    weights: torch.Tensor,
    k_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    num_tokens = query.shape[0]
    page_rows = k_pool.shape[1]
    keys_per_page = KEYS_PER_WORD_ROW * page_rows
    pool_keys = _unpack_split_bf16_keys(k_pool).view(
        k_pool.shape[0], keys_per_page, INDEX_HEAD_DIM
    )
    out = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=query.device)
    for s_idx in range(seq_lens.shape[0]):
        start = int(cu_q_lens[s_idx].item())
        end = int(cu_q_lens[s_idx + 1].item())
        q_len = end - start
        if q_len <= 0:
            continue
        seq_len = int(seq_lens[s_idx].item())
        ctx_base = seq_len - q_len
        for row_i in range(start, end):
            pos = ctx_base + (row_i - start)
            num_valid = (pos + 1) // compress_ratio
            if candidate_indices is not None:
                cands = candidate_indices[row_i].tolist()
                valid_pos = [c for c in cands if 0 <= c < num_valid]
            else:
                valid_pos = list(range(num_valid))
            if not valid_pos:
                continue
            scored: list[tuple[float, int]] = []
            q_row = query[row_i].float()
            w_row = weights[row_i].float().squeeze(-1)
            for kp in valid_pos:
                page_col = kp // keys_per_page
                in_page = kp % keys_per_page
                phys_page = int(block_tables[s_idx, page_col].item())
                k_vec = pool_keys[phys_page, in_page]
                score = float((torch.relu(q_row @ k_vec) * w_row).sum().item())
                scored.append((score, kp))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            chosen = [kp for _, kp in scored[:topk]]
            out[row_i, : len(chosen)] = torch.tensor(
                chosen, dtype=torch.int32, device=query.device
            )
    return out


def _ref_gather_sparse_fp8_kv(
    kv_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    topk_indices: torch.Tensor,
    block_tables: torch.Tensor,
    cu_q_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv_pool.ndim == 3 and rope_pool.ndim == 3
    assert kv_pool.shape[1] == page_size
    assert page_size % 4 == 0 and rope_pool.shape[1] == page_size // 4
    num_tokens, topk = topk_indices.shape
    nope_out = torch.zeros(
        (num_tokens, topk, ROW_WORDS), dtype=torch.int32, device=kv_pool.device
    )
    rope_out = torch.zeros(
        (num_tokens, topk // 4, ROW_WORDS), dtype=torch.int32, device=rope_pool.device
    )
    num_seqs, table_width = block_tables.shape
    rope_flat = rope_pool.reshape(-1, ROPE_WORDS)
    for s_idx in range(num_seqs):
        start = int(cu_q_lens[s_idx].item())
        end = int(cu_q_lens[s_idx + 1].item())
        for tok in range(start, end):
            for k in range(topk):
                idx = int(topk_indices[tok, k].item())
                if idx < 0:
                    continue
                col = idx // page_size
                if col >= table_width:
                    continue
                phys_page = int(block_tables[s_idx, col].item())
                if phys_page < 0 or phys_page >= kv_pool.shape[0]:
                    continue
                in_page = idx % page_size
                nope_out[tok, k] = kv_pool[phys_page, in_page]
                q_row = k // 4
                q_off = (k % 4) * ROPE_WORDS
                phys_slot = phys_page * page_size + in_page
                rope_out[tok, q_row, q_off : q_off + ROPE_WORDS] = rope_flat[phys_slot]
    return nope_out, rope_out


def _ref_sparse_mla_swa_attention(
    q: torch.Tensor,
    gathered_nope: torch.Tensor,
    gathered_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    attention_sinks: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    sm_scale: float,
    swa_window: int,
    swa_kv: torch.Tensor | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = q.shape
    num_seqs = seq_lens.shape[0]
    out = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
    if num_seqs == 0:
        return out
    swa_width = (
        0 if swa_kv is None else (swa_window if swa_kv.ndim == 2 else swa_kv.shape[1])
    )
    for s_idx in range(num_seqs):
        q_start = int(cu_q_lens[s_idx].item())
        q_end = int(cu_q_lens[s_idx + 1].item())
        if q_end <= q_start:
            continue
        seq_len = int(seq_lens[s_idx].item())
        min_pos = 0 if swa_min_pos is None else max(0, int(swa_min_pos[s_idx].item()))
        for tok in range(q_start, q_end):
            pos = seq_len - (q_end - tok)
            keys: list[torch.Tensor] = []
            if swa_kv is not None and swa_window > 0 and swa_width > 0:
                if swa_kv.ndim == 2:
                    delta = tok - q_start
                    prefix_rows = swa_kv[: num_seqs * swa_width]
                    current_rows = swa_kv[num_seqs * swa_width :]
                    base_pos = pos - swa_width + 1
                    hist_keep = max(0, swa_width - (delta + 1))
                    seq_prefix = prefix_rows[
                        s_idx * swa_width + (swa_width - hist_keep) : (s_idx + 1)
                        * swa_width
                    ]
                    curr_start = max(q_start, tok - swa_width + 1)
                    seq_curr = current_rows[curr_start : tok + 1]
                    win_keys = torch.cat((seq_prefix, seq_curr), dim=0)
                    win_pos = torch.arange(base_pos, pos + 1, dtype=torch.int64)
                    if win_keys.shape[0] < win_pos.shape[0]:
                        pad_n = win_pos.shape[0] - win_keys.shape[0]
                        win_keys = torch.cat(
                            (
                                torch.zeros((pad_n, head_dim), dtype=win_keys.dtype),
                                win_keys,
                            ),
                            dim=0,
                        )
                    valid_swa = (win_pos >= min_pos) & (win_pos <= pos)
                else:
                    base_pos = max(0, pos - swa_width + 1)
                    win_keys = swa_kv[tok]
                    win_pos = torch.arange(
                        base_pos, base_pos + swa_width, dtype=torch.int64
                    )
                    valid_swa = (win_pos >= min_pos) & (win_pos <= pos)
                if valid_swa.any():
                    keys.append(win_keys[valid_swa].float())
            max_comp = (pos + 1) // kv_compress_ratio
            for k_idx in range(topk_indices.shape[1]):
                c_idx = int(topk_indices[tok, k_idx].item())
                if 0 <= c_idx < max_comp:
                    q_row = k_idx // 4
                    q_off = (k_idx % 4) * ROPE_WORDS
                    keys.append(
                        _dequantize_packed_fp8_row(
                            gathered_nope[tok, k_idx],
                            gathered_rope[tok, q_row, q_off : q_off + ROPE_WORDS],
                        ).unsqueeze(0)
                    )
            if keys:
                all_k = torch.cat(keys, dim=0)
                logits = (q[tok].float() @ all_k.T) * sm_scale
                all_logits = torch.cat(
                    (logits, attention_sinks.float().unsqueeze(1)), dim=1
                )
                probs = torch.softmax(all_logits, dim=1)
                out[tok] = (probs[:, :-1] @ all_k).to(torch.bfloat16)
    return out


class _RunnerTestHarness(TPUModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        max_num_reqs: int = 4,
        max_model_len: int = 256,
    ) -> None:
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = None
        self.device = device
        self.kv_cache_dtype = torch.bfloat16
        self._unified_kv_layout = False
        self._mamba_num_blocks = None
        self._uniform_mamba_layout = False
        self.use_spmd = False
        self.enforce_eager = True
        self.max_num_reqs = max_num_reqs
        self.most_model_len = max_model_len
        self.kv_cache_config = None
        self.kv_caches = []
        self.kv_cache_raw_tensors = []
        self.shared_kv_cache_layers = {}
        self.runner_only_attn_layers = set()
        self.block_table_cpu = torch.zeros((max_num_reqs, 8), dtype=torch.int32)
        self.attn_groups = []
        self.ds_v41_zero_owner = None
        self._pool_block_split = 1
        self._mamba_copy_plan = []
        self.bm_pool_layout = None
        self._registered_h2d_sources = set()
        self._free_h2d_sources = {}
        self._unfenced_h2d_sources = []
        self.position_ids = torch.zeros(16, dtype=torch.int32, device=device)
        self.input_batch = cast(Any, SimpleNamespace(block_table=[]))
        self._precompile_dense_kv_cache_block_copies = MagicMock()
        self._initialize_attention_kernels = MagicMock()
        self._precompile_substitute_placeholder_token = MagicMock()
        self._precompile_mamba_state_seed_copies = MagicMock()

    def may_reinitialize_input_batch(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        del kernel_block_sizes
        tables = [
            SimpleNamespace(
                max_num_blocks_per_req=4,
                get_cpu_tensor=lambda: torch.zeros(
                    (self.max_num_reqs, 4), dtype=torch.int32
                ),
            )
            for _ in kv_cache_config.kv_cache_groups
        ]
        self.input_batch = cast(Any, SimpleNamespace(block_table=tables))

    def zero_blocks_public(self, block_ids: list[int]) -> None:
        self._zero_block_ids(block_ids)


@pytest.fixture
def stub_dsv41_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, list[Any]]]:
    trace: dict[str, list[Any]] = {
        "write_sub_rows": [],
        "gather_swa_windows": [],
        "gather_sparse_fp8_kv": [],
        "sparse_mla_swa_attention": [],
    }

    def _write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        *,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        del page_tokens
        trace["write_sub_rows"].append({"bank": bank, "after": after})
        out = bank.clone()
        sub = out.view(-1, updates.shape[1])
        loc = slots.long()
        valid = (loc >= 0) & (loc < sub.shape[0])
        sub[loc[valid]] = updates[valid]
        return out

    def _gather_rows(pool: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        pool_2d = pool.reshape(-1, ROW_WORDS)
        valid = (idx >= 0) & (idx < pool_2d.shape[0])
        safe = idx.masked_fill(~valid, 0).long()
        return pool_2d.index_select(0, safe).masked_fill(~valid.unsqueeze(-1), 0)

    def _gather_swa(
        pool: torch.Tensor,
        page_ids: torch.Tensor,
        phase_rows: torch.Tensor,
        *,
        window_rows: int,
        rows_per_page: int,
        page_row_stride: int,
        page_row_offset: int = 0,
    ) -> torch.Tensor:
        trace["gather_swa_windows"].append(
            {
                "pool": pool,
                "page_ids": page_ids.clone(),
                "phase_rows": phase_rows.clone(),
                "window_rows": window_rows,
            }
        )
        # Mirror Pallas kernel row-granular pages_per_window validation.
        min_pages = -(-(window_rows - 1) // rows_per_page) + 1
        if page_ids.shape[1] < min_pages:
            raise ValueError("pages_per_window too small to cover every window phase.")
        pool_2d = pool.reshape(-1, ROW_WORDS)
        num_seqs = page_ids.shape[0]
        win_iota = torch.arange(window_rows, dtype=torch.int64, device=pool.device)
        rel_rows = phase_rows.to(torch.int64)[:, None] + win_iota[None, :]
        p_col = rel_rows // rows_per_page
        in_page = rel_rows % rows_per_page
        pids = page_ids.to(torch.int64).gather(1, p_col)
        valid = pids >= 0
        phys = (pids.clamp_min(0) * page_row_stride + page_row_offset + in_page).clamp(
            0, pool_2d.shape[0] - 1
        )
        out = pool_2d[phys.reshape(-1)].reshape(num_seqs, window_rows, ROW_WORDS)
        # Emulate hardware "unspecified on page_id < 0" with BF16 NaN bit pattern.
        return out.masked_fill(~valid.unsqueeze(-1), 0x7FC07FC0)

    def _gather_sparse(
        kv_pool: torch.Tensor,
        rope_pool: torch.Tensor,
        topk_indices: torch.Tensor,
        block_tables: torch.Tensor,
        cu_q_lens: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace["gather_sparse_fp8_kv"].append(
            {
                "kv_pool": kv_pool,
                "rope_pool": rope_pool,
                "topk_indices": topk_indices.clone(),
                "block_tables": block_tables.clone(),
                "cu_q_lens": cu_q_lens.clone(),
                "page_size": page_size,
            }
        )
        return _ref_gather_sparse_fp8_kv(
            kv_pool, rope_pool, topk_indices, block_tables, cu_q_lens, page_size
        )

    def _mla_swa(
        q: torch.Tensor,
        gathered_nope: torch.Tensor,
        gathered_rope: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_sinks: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        sm_scale: float,
        swa_window: int,
        swa_kv: torch.Tensor | None = None,
        kv_compress_ratio: int = 1,
        swa_min_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trace["sparse_mla_swa_attention"].append(
            {
                "q": q.clone(),
                "gathered_nope": gathered_nope.clone(),
                "gathered_rope": gathered_rope.clone(),
                "topk_indices": topk_indices.clone(),
                "seq_lens": seq_lens.clone(),
                "cu_q_lens": cu_q_lens.clone(),
                "swa_window": swa_window,
                "swa_kv": None if swa_kv is None else swa_kv.clone(),
                "kv_compress_ratio": kv_compress_ratio,
                "swa_min_pos": None if swa_min_pos is None else swa_min_pos.clone(),
            }
        )
        return _ref_sparse_mla_swa_attention(
            q,
            gathered_nope,
            gathered_rope,
            topk_indices,
            attention_sinks,
            seq_lens,
            cu_q_lens,
            sm_scale,
            swa_window,
            swa_kv,
            kv_compress_ratio,
            swa_min_pos,
        )

    for cls in (DeepseekCompressor, DeepseekV41Indexer, DeepseekV41Attention):
        monkeypatch.setattr(cls, "kv_rope_op", property(lambda _s: _cpu_rope))
    monkeypatch.setattr(
        DeepseekV41Attention, "o_proj_op", property(lambda _s: _cpu_inverse_rope_wo_a)
    )
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _write)
    monkeypatch.setattr(
        cache_mod.cache_op, "dsv41_pack_fp8_kv_rows", _ref_pack_fp8_kv_rows
    )
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_gather_cache_rows", _gather_rows)
    monkeypatch.setattr(
        cache_mod.swa_window_op, "dsv41_gather_swa_windows", _gather_swa
    )
    monkeypatch.setattr(
        indexer_mod.indexer_op, "dsv41_paged_indexer_topk", _ref_paged_indexer_topk
    )
    monkeypatch.setattr(
        dsv41_ops.attention_op, "dsv41_gather_sparse_fp8_kv", _gather_sparse
    )
    monkeypatch.setattr(
        dsv41_ops.attention_op, "dsv41_sparse_mla_swa_attention", _mla_swa
    )
    yield trace


def _reference_attention_fp32(
    q_bf16: torch.Tensor,
    swa_history_by_seq: list[torch.Tensor],
    comp_history_by_seq: list[torch.Tensor] | None,
    topk_indices: torch.Tensor | None,
    attn_sink: torch.Tensor,
    seq_lens: list[int],
    cu_q_lens: list[int],
    *,
    sm_scale: float,
    window_size: int,
    compress_ratio: int = 1,
    min_pos_by_seq: list[int] | None = None,
    non_causal: bool = False,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = q_bf16.shape
    out = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.float32)
    for s_idx in range(len(seq_lens)):
        q_start, q_end = cu_q_lens[s_idx], cu_q_lens[s_idx + 1]
        seq_len = seq_lens[s_idx]
        min_pos = 0 if min_pos_by_seq is None else max(0, min_pos_by_seq[s_idx])
        swa_hist = swa_history_by_seq[s_idx].float()
        for t in range(q_start, q_end):
            pos = seq_len - 1 if non_causal else seq_len - (q_end - t)
            w_start = max(min_pos, pos - window_size + 1)
            swa_keys = swa_hist[w_start : pos + 1]
            keys = [swa_keys]
            if comp_history_by_seq is not None and topk_indices is not None:
                comp_hist = comp_history_by_seq[s_idx].float()
                max_comp = (pos + 1) // compress_ratio
                t_indices = [
                    int(idx)
                    for idx in topk_indices[t].tolist()
                    if 0 <= int(idx) < max_comp
                ]
                if t_indices:
                    keys.append(comp_hist[t_indices])
            all_k = torch.cat(keys, dim=0)
            q_vec = q_bf16[t].float()
            logits = (q_vec @ all_k.T) * sm_scale
            all_logits = torch.cat((logits, attn_sink.float().unsqueeze(1)), dim=1)
            probs = torch.softmax(all_logits, dim=1)
            out[t] = probs[:, :-1] @ all_k
    return out


def test_should_keep_attention_module_decoupled_from_v4_private_apis_and_two_bank() -> (
    None
):
    source = inspect.getsource(attn_mod)
    tree = ast.parse(source)
    modules: set[str] = set()
    gather_arg_counts: list[int] = []
    banned_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "dsv41_gather_sparse_fp8_kv"
            ):
                gather_arg_counts.append(len(node.args) + len(node.keywords))
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr", "vars")
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "gcd"):
                banned_calls.append(ast.unparse(node.func))
    assert not any(
        "deepseek_v4." in mod or mod.endswith("deepseek_v4") for mod in modules
    )
    assert not any("torch_tpu._internal" in mod for mod in modules)
    assert "SLF001" not in source
    assert gather_arg_counts == [6]
    assert banned_calls == []


def test_should_build_self_contained_o_proj_op_and_enforce_weight_guards(
    cpu_dist_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    vcfg = _make_dsv41_vllm_config(block_size=16, max_model_len=64)
    with set_current_vllm_config(vcfg):
        layer = DeepseekV41Attention(vcfg, "model.layers.0.attn")

    # process_weights_after_loading: transpose -> idempotent -> reject bad shape.
    _init_attention_weights(layer, seed=5, fp8_wo_a=True)
    orig_w = cast(torch.Tensor, layer.wo_a.weight).detach().clone()
    layer.process_weights_after_loading(torch.bfloat16)
    expected_shape = (NUM_HEADS * MAIN_HEAD_DIM // O_GROUPS, O_GROUPS * O_LORA_RANK)
    transposed_w = cast(torch.Tensor, layer.wo_a.weight)
    assert tuple(transposed_w.shape) == expected_shape
    assert torch.equal(transposed_w, orig_w.t())
    layer.process_weights_after_loading(torch.bfloat16)
    assert tuple(cast(torch.Tensor, layer.wo_a.weight).shape) == expected_shape

    saved_w = cast(nn.Parameter, layer.wo_a.weight)
    layer.wo_a.weight = nn.Parameter(
        torch.empty((3, 5), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    with pytest.raises(ValueError, match="expected weight shape"):
        layer.process_weights_after_loading(torch.bfloat16)
    layer.wo_a.weight = saved_w

    # _assert_o_proj_supported fail-fast guards via public o_proj_op property.
    layer.wo_a.weight = nn.Parameter(saved_w.to(torch.bfloat16), requires_grad=False)
    with pytest.raises(ValueError, match="fp8_e4m3fn"):
        _ = layer.o_proj_op
    layer.wo_a.weight = saved_w

    saved_scale = layer.wo_a.weight_scale
    del layer.wo_a.weight_scale
    with pytest.raises(ValueError, match="weight_scale"):
        _ = layer.o_proj_op
    layer.wo_a.weight_scale = nn.Parameter(
        torch.ones((2, O_LORA_RANK), dtype=torch.float32), requires_grad=False
    )
    with pytest.raises(ValueError, match="weight_scale"):
        _ = layer.o_proj_op
    layer.wo_a.weight_scale = saved_scale

    layer.n_local_heads = 4
    with pytest.raises(ValueError, match="8 heads per group"):
        _ = layer.o_proj_op
    layer.n_local_heads = NUM_HEADS

    layer.head_dim = 64
    with pytest.raises(ValueError, match="multiple of 128"):
        _ = layer.o_proj_op
    layer.head_dim = MAIN_HEAD_DIM

    saved_cs = layer.rotary_emb.cos_sin_cache
    layer.rotary_emb.cos_sin_cache = torch.zeros((16, 130), dtype=torch.float32)
    with pytest.raises(ValueError, match="rotary_dim"):
        _ = layer.o_proj_op
    layer.rotary_emb.cos_sin_cache = saved_cs

    # Validate public kv_rope_op and self-contained o_proj_op registration, mesh
    # cache isolation, FakeTensorMode shape inference, and JAX bridge forwarding.
    recorded: list[dict[str, Any]] = []
    orig_jax_op = tpu_pallas.jax_op

    def _spy_jax_op(name: str, fn: Any, **kwargs: Any) -> Any:
        recorded.append({"name": name, "mesh": kwargs.get("mesh"), **kwargs})
        return orig_jax_op(name, fn, **kwargs)

    def _ref_o_proj_jax(
        x: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        wo_a: jax.Array,
        wo_a_scale: jax.Array,
        *,
        head_dim: int,
        inverse: bool,
        quantize_activations: bool,
    ) -> jax.Array:
        assert head_dim == MAIN_HEAD_DIM and inverse and quantize_activations
        pos_scale = cos_sin_cache[positions, :1]
        base = jnp.sum(x.astype(jnp.float32), axis=(1, 2), keepdims=False)[:, None]
        return (
            (base + pos_scale) * wo_a_scale.reshape(1, -1)
            + wo_a[:1, :].astype(jnp.float32)
        ).astype(x.dtype)

    monkeypatch.setattr(attn_mod.pallas, "jax_op", _spy_jax_op)
    monkeypatch.setattr(attn_mod, "fused_reverse_rope_wo_a_projection", _ref_o_proj_jax)

    single_mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    with set_vllm_model_wrapper_context(mesh=single_mesh):
        assert layer.kv_rope_op is attn_mod.get_kv_rope_op()
        op_single = layer.o_proj_op
        assert layer.o_proj_op is op_single
        o_proj_records = [r for r in recorded if "dsv41_o_proj" in r["name"]]
        assert len(o_proj_records) == 1
        assert o_proj_records[-1]["mesh"] is None
        assert o_proj_records[-1]["input_partition_specs"] is None

        x_np = np.ones((2, NUM_HEADS, MAIN_HEAD_DIM), dtype=np.float32)
        pos_np = np.array([1, 3], dtype=np.int32)
        cs_np = np.arange(8 * ROPE_HEAD_DIM, dtype=np.float32).reshape(8, ROPE_HEAD_DIM)
        wo_a_np = np.ones(expected_shape, dtype=np.float32)
        scale_np = np.arange(O_LORA_RANK, dtype=np.float32) * 0.1
        out_jax = op_single(
            _JaxBackedTensor(jnp.asarray(x_np, dtype=jnp.bfloat16)),
            _JaxBackedTensor(jnp.asarray(pos_np, dtype=jnp.int32)),
            _JaxBackedTensor(jnp.asarray(cs_np, dtype=jnp.float32)),
            _JaxBackedTensor(jnp.asarray(wo_a_np, dtype=jnp.float8_e4m3fn)),
            _JaxBackedTensor(jnp.asarray(scale_np, dtype=jnp.float32)),
        )
        assert isinstance(out_jax, _JaxBackedTensor)
        expected = _ref_o_proj_jax(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(pos_np, dtype=jnp.int32),
            jnp.asarray(cs_np, dtype=jnp.float32),
            jnp.asarray(wo_a_np, dtype=jnp.float8_e4m3fn),
            jnp.asarray(scale_np, dtype=jnp.float32),
            head_dim=MAIN_HEAD_DIM,
            inverse=True,
            quantize_activations=True,
        )
        np.testing.assert_array_equal(
            np.asarray(out_jax.to_numpy(), dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
        )

        counter = CompileCounterWithBackend("aot_eager")
        compiled = torch.compile(
            lambda x, pos, cs, w, s: op_single(x, pos, cs, w, s),
            backend=counter,
            fullgraph=True,
        )
        with FakeTensorMode():
            fake_out = compiled(
                torch.empty((4, NUM_HEADS, MAIN_HEAD_DIM), dtype=torch.bfloat16),
                torch.empty((4,), dtype=torch.int32),
                torch.empty((16, ROPE_HEAD_DIM), dtype=torch.float32),
                torch.empty(expected_shape, dtype=torch.float8_e4m3fn),
                torch.empty((O_LORA_RANK,), dtype=torch.float32),
            )
            assert fake_out.shape == (4, O_LORA_RANK)
            assert fake_out.dtype == torch.bfloat16

    multi_mesh = AbstractMesh((2,), ("model",))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with set_vllm_model_wrapper_context(mesh=cast(Any, multi_mesh)):
        op_multi = layer.o_proj_op
        assert layer.o_proj_op is op_multi
        assert op_multi is not op_single
        o_proj_records = [r for r in recorded if "dsv41_o_proj" in r["name"]]
        assert len(o_proj_records) == 2
        assert o_proj_records[-1]["mesh"] is multi_mesh
        assert o_proj_records[-1]["input_partition_specs"] == (
            P(None, "model", None),
            P(),
            P(),
            P(),
            P(),
        )

    # _o_proj fail-fast dtype/shape checks, position clamping, and tuple wo_b.
    seen_o_proj_positions: list[torch.Tensor] = []

    def _spy_inverse_rope_wo_a(
        attn_out: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        wo_a: torch.Tensor,
        wo_a_scale: torch.Tensor,
    ) -> torch.Tensor:
        seen_o_proj_positions.append(positions.detach().clone())
        return _cpu_inverse_rope_wo_a(
            attn_out, positions, cos_sin_cache, wo_a, wo_a_scale
        )

    monkeypatch.setattr(
        DeepseekV41Attention, "kv_rope_op", property(lambda _s: _cpu_rope)
    )
    monkeypatch.setattr(
        DeepseekV41Attention,
        "o_proj_op",
        property(lambda _s: _spy_inverse_rope_wo_a),
    )
    pos_4 = torch.arange(4, dtype=torch.int32)
    pos_oob = torch.tensor([-5, 0, 1, 99999], dtype=torch.int32)
    h_bf16 = torch.randn((4, HIDDEN_SIZE), dtype=torch.bfloat16)

    # Tuple-returning wo_b + OOB position clamping via public forward().
    orig_wo_b = layer.wo_b

    class _TupleWoB(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return cast(torch.Tensor, orig_wo_b(x)), None

    layer.wo_b = cast(Any, _TupleWoB())
    with override_forward_context(_forward_ctx(None)):
        tuple_out = layer(pos_oob, h_bf16)
    assert tuple_out.shape == (4, HIDDEN_SIZE)
    max_pos = int(cast(torch.Tensor, layer.rotary_emb.cos_sin_cache).shape[0]) - 1
    assert torch.equal(
        seen_o_proj_positions[-1],
        torch.tensor([0, 0, 1, max_pos], dtype=torch.int32),
    )
    layer.wo_b = orig_wo_b

    # Wrong attn_out shape via padded_heads != n_local_heads in public forward().
    layer.padded_heads = NUM_HEADS * 2
    with (
        override_forward_context(_forward_ctx(None)),
        pytest.raises(ValueError, match="requires shape"),
    ):
        layer(pos_4, h_bf16)
    layer.padded_heads = NUM_HEADS

    # Wrong attn_out dtype (fp32) via fp32 model config in public forward().
    vcfg_f32 = _make_dsv41_vllm_config(
        block_size=16, model_dtype=torch.float32, max_model_len=64
    )
    with set_current_vllm_config(vcfg_f32):
        layer_f32 = DeepseekV41Attention(vcfg_f32, "model.layers.0.attn")
    with (
        override_forward_context(_forward_ctx(None)),
        pytest.raises(ValueError, match="bf16 activations"),
    ):
        layer_f32(pos_4, torch.randn((4, HIDDEN_SIZE), dtype=torch.float32))


def test_should_reject_invalid_init_and_report_specs_when_configuring_swa_and_attention(
    cpu_dist_ctx: None,
) -> None:
    del cpu_dist_ctx
    vcfg_fp8 = _make_dsv41_vllm_config(
        block_size=32, cache_dtype="fp8_ds_mla", max_model_len=128
    )
    with set_current_vllm_config(vcfg_fp8):
        with pytest.raises(ValueError, match="bfloat16"):
            SWACache(
                head_dim=MAIN_HEAD_DIM,
                window_size=SLIDING_WINDOW,
                dtype=torch.uint8,
                prefix="model.layers.0.attn.bad_swa",
                cache_config=vcfg_fp8.cache_config,
            )
        swa = SWACache(
            head_dim=MAIN_HEAD_DIM,
            window_size=SLIDING_WINDOW,
            dtype=torch.bfloat16,
            prefix="model.layers.0.attn.swa_cache",
            cache_config=vcfg_fp8.cache_config,
            block_size=32,
        )
        assert isinstance(swa, DsV41CacheMixin)
        assert swa.get_attn_backend() is AttentionBackend
        swa_spec = swa.get_kv_cache_spec(vcfg_fp8)
        assert isinstance(swa_spec, SlidingWindowMLASpec)
        assert swa_spec.dtype == torch.bfloat16
        assert swa_spec.alignment == ROW_BYTES == KV_PAGE_ALIGNMENT
        assert swa_spec.state_content_bytes is None
        assert swa_spec.page_size_padded is None
        assert swa_spec.block_size == 32
        assert PACKED_ROW_BYTES == 584

    vcfg_missing_cache = _make_dsv41_vllm_config(block_size=16)
    vcfg_missing_cache.cache_config = cast(Any, None)
    with (
        set_current_vllm_config(vcfg_missing_cache),
        pytest.raises(ValueError, match="cache_config"),
    ):
        DeepseekV41Attention(vcfg_missing_cache, "model.layers.0.attn")

    vcfg_bad_ratio = _make_dsv41_vllm_config(compress_ratios=[4])
    with (
        set_current_vllm_config(vcfg_bad_ratio),
        pytest.raises(ValueError, match="compress_ratio"),
    ):
        DeepseekV41Attention(vcfg_bad_ratio, "model.layers.0.attn")

    vcfg_missing_k = _make_dsv41_vllm_config(
        compress_ratios=[0, 0, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2, 3],
    )
    with (
        set_current_vllm_config(vcfg_missing_k),
        pytest.raises(NotImplementedError, match="indexer.k_cache"),
    ):
        DeepseekV41Attention(vcfg_missing_k, "model.layers.3.attn")

    vcfg_empty_kv_src = _make_dsv41_vllm_config(
        compress_ratios=[0, 2],
        kv_source_layer_ids=[],
        index_source_layer_ids=[1],
    )
    with (
        set_current_vllm_config(vcfg_empty_kv_src),
        pytest.raises(ValueError, match="kv_source_layer_ids"),
    ):
        DeepseekV41Attention(vcfg_empty_kv_src, "model.layers.1.attn")

    vcfg_empty_idx_src = _make_dsv41_vllm_config(
        compress_ratios=[0, 2],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[],
    )
    with (
        set_current_vllm_config(vcfg_empty_idx_src),
        pytest.raises(ValueError, match="index_source_layer_ids"),
    ):
        DeepseekV41Attention(vcfg_empty_idx_src, "model.layers.1.attn")

    vcfg_no_prior_kv = _make_dsv41_vllm_config(
        compress_ratios=[0, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
    )
    with (
        set_current_vllm_config(vcfg_no_prior_kv),
        pytest.raises(ValueError, match="kv_source_layer_id"),
    ):
        DeepseekV41Attention(vcfg_no_prior_kv, "model.layers.1.attn")

    vcfg_no_prior_idx = _make_dsv41_vllm_config(
        compress_ratios=[0, 2, 2],
        kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[2],
    )
    with (
        set_current_vllm_config(vcfg_no_prior_idx),
        pytest.raises(ValueError, match="index_source_layer_id"),
    ):
        DeepseekV41Attention(vcfg_no_prior_idx, "model.layers.1.attn")

    vcfg_orphan_follower = _make_dsv41_vllm_config(
        compress_ratios=[0, 0, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
    )
    with (
        set_current_vllm_config(vcfg_orphan_follower),
        pytest.raises(NotImplementedError, match="Compressed-KV source"),
    ):
        DeepseekV41Attention(vcfg_orphan_follower, "model.layers.3.attn")

    vcfg_dup = _make_dsv41_vllm_config(block_size=16)
    with set_current_vllm_config(vcfg_dup):
        DeepseekV41Attention(vcfg_dup, "model.layers.0.attn")
        with pytest.raises(
            ValueError, match=r"Duplicate layer name: model\.layers\.0\.attn$"
        ):
            DeepseekV41Attention(vcfg_dup, "model.layers.0.attn")

    vcfg_trunc = _make_dsv41_vllm_config(
        block_size=16, max_model_len=64, max_position_embeddings=256
    )
    vcfg_trunc.cache_config.block_size = cast(Any, None)
    vcfg_exact = _make_dsv41_vllm_config(
        block_size=16, max_model_len=256, max_position_embeddings=128
    )
    with set_current_vllm_config(vcfg_trunc):
        l0_trunc = DeepseekV41Attention(vcfg_trunc, "model.layers.0.attn")
        l40_draft = DeepseekV41Attention(vcfg_trunc, "model.layers.40.attn")
        assert l0_trunc.rotary_emb.cos_sin_cache.shape[0] == 64
        shared_rope = build_deepseek_v4_rope(
            vcfg_trunc.model_config.hf_config,
            head_dim=MAIN_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            max_position_embeddings=256,
            compress_ratio=0,
        )
        assert shared_rope.cos_sin_cache.shape == (256, ROPE_HEAD_DIM)
        assert l0_trunc.rotary_emb is not shared_rope
        assert l0_trunc.swa_cache_layer.block_size == 64
        assert l0_trunc.get_kv_cache_spec(vcfg_trunc) is None
        assert l40_draft.compress_ratio == 0
        assert l40_draft.get_kv_cache_spec(vcfg_trunc) is None
        assert l0_trunc.get_attn_backend() is AttentionBackend
        assert DeepseekV41Attention.get_padded_num_q_heads(NUM_HEADS) == NUM_HEADS

    with set_current_vllm_config(vcfg_exact):
        l2_src = DeepseekV41Attention(vcfg_exact, "model.layers.2.attn")
        l3_fol = DeepseekV41Attention(vcfg_exact, "model.layers.3.attn")
        assert l2_src.rotary_emb.cos_sin_cache.shape[0] == 128
        spec_l2 = l2_src.get_kv_cache_spec(vcfg_exact)
        assert isinstance(spec_l2, MLAAttentionSpec)
        assert spec_l2.dtype == torch.uint8
        assert spec_l2.alignment == ROW_BYTES
        assert spec_l2.state_content_bytes == ROW_BYTES + ROPE_HEAD_DIM * 2
        assert spec_l2.page_size_padded is None
        assert spec_l2.tokens_per_state == 2
        assert l3_fol.get_kv_cache_spec(vcfg_exact) is None


def test_should_route_exclusively_to_dsv41_allocator_when_registered(
    cpu_dist_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    dsv4_calls = 0

    def _forbid_dsv4(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dsv4_calls
        dsv4_calls += 1
        raise AssertionError("DsV4KVCacheAllocator must not run for DSv4.1 layers")

    for attr_name, attr_val in vars(kcm_mod.DsV4KVCacheAllocator).items():
        if attr_name != "__init__" and callable(attr_val):
            monkeypatch.setattr(kcm_mod.DsV4KVCacheAllocator, attr_name, _forbid_dsv4)

    vcfg = _make_dsv41_vllm_config(block_size=16, max_model_len=256, max_num_seqs=4)
    topk_buf = torch.full((16, 8), -1, dtype=torch.int32)
    cand_buf = torch.full((16, 4), -1, dtype=torch.int32)
    with set_current_vllm_config(vcfg):
        l0 = DeepseekV41Attention(vcfg, "model.layers.0.attn", topk_buf)
        l2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_buf, None, cand_buf)
        l3 = DeepseekV41Attention(vcfg, "model.layers.3.attn", topk_buf, None, cand_buf)
        l8 = DeepseekV41Attention(vcfg, "model.layers.8.attn", topk_buf, None, cand_buf)
        l14 = DeepseekV41Attention(
            vcfg, "model.layers.14.attn", topk_buf, None, cand_buf
        )
        l19 = DeepseekV41Attention(
            vcfg, "model.layers.19.attn", topk_buf, None, cand_buf
        )
        l20 = DeepseekV41Attention(
            vcfg, "model.layers.20.attn", topk_buf, None, cand_buf
        )

    assert l19.indexer is not None and l20.indexer is not None
    assert l19.indexer.indexer_op.candidate_write is True
    assert l20.indexer.indexer_op.candidate_write is False

    runner = _RunnerTestHarness(vcfg, torch.device("cpu"), max_num_reqs=4)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    assert "model.layers.0.attn" not in specs
    assert "model.layers.3.attn" not in specs
    assert "model.layers.19.attn" not in specs
    for src_name in (
        "model.layers.2.attn",
        "model.layers.8.attn",
        "model.layers.14.attn",
        "model.layers.20.attn",
        "model.layers.0.attn.swa_cache",
        "model.layers.2.attn.swa_cache",
        "model.layers.2.attn.indexer.k_cache",
        "model.layers.2.attn.compressor.state_cache",
    ):
        assert src_name in specs

    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 6)
    mgr.initialize_kv_cache(kv_cfg)

    assert dsv4_calls == 0
    for layer in (l2, l8, l14, l20):
        assert isinstance(layer.paged_cache_rows, PagedCacheRows)
        assert layer.paged_cache_rows.null_block is True
    for layer in (l0, l2, l3, l8, l14, l19, l20):
        assert isinstance(layer.swa_cache_layer.paged_cache_rows, PagedCacheRows)
        assert layer.swa_cache_layer.paged_cache_rows.null_block is True
        assert layer.swa_cache_layer.paged_cache_rows.supports_paged_windows is True

    assert l20.paged_cache_rows is not None
    assert l14.paged_cache_rows is not None
    assert l2.paged_cache_rows is not None
    assert l8.paged_cache_rows is not None
    assert l20.paged_cache_rows.page_stride == l20.paged_cache_rows.shape[1]
    assert l14.paged_cache_rows.page_stride == l14.paged_cache_rows.shape[1]
    assert l2.paged_cache_rows.page_stride == 2 * l2.paged_cache_rows.shape[1]
    assert l8.paged_cache_rows.page_stride == 2 * l8.paged_cache_rows.shape[1]
    assert l2.paged_cache_rows.raw is l8.paged_cache_rows.raw
    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].abs().max().item()) == 0


def test_should_dispatch_paged_swa_windows_and_mask_nans_across_swa_layouts(
    cpu_dist_ctx: None,
    stub_dsv41_ops: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    vcfg = _make_dsv41_vllm_config(
        block_size=4, sliding_window=6, max_model_len=64, max_position_embeddings=64
    )
    with set_current_vllm_config(vcfg):
        layer = DeepseekV41Attention(vcfg, "model.layers.0.attn")
    _init_attention_weights(layer, seed=10)

    q = torch.randn((4, NUM_HEADS, MAIN_HEAD_DIM), dtype=torch.bfloat16) * 0.1
    kv = torch.randn((4, MAIN_HEAD_DIM), dtype=torch.bfloat16) * 0.1
    pos = torch.tensor([6, 7, 2, 3], dtype=torch.int32)
    out = torch.ones_like(q)
    with override_forward_context(_forward_ctx(None)):
        layer.forward_mqa(q, kv, pos, out)
    assert int(out.abs().max().item()) == 0

    num_blocks, page_size = 5, 4
    rows_per_tok = 2
    gen = torch.Generator().manual_seed(77)
    logical_bf16 = torch.randn(
        (num_blocks, page_size, MAIN_HEAD_DIM), generator=gen, dtype=torch.bfloat16
    )
    logical_bf16[0].fill_(float("nan"))
    logical_bf16[4].fill_(float("nan"))

    flat = logical_bf16.reshape(-1, MAIN_HEAD_DIM)
    even = (flat[:, 0::2].float().view(torch.int32) >> 16) & 0xFFFF
    odd = flat[:, 1::2].float().view(torch.int32) & (-65536)
    raw_3d = (even | odd).reshape(num_blocks, page_size * rows_per_tok, ROW_WORDS)
    layout = (
        (num_blocks, page_size, MAIN_HEAD_DIM),
        torch.bfloat16,
        page_size * MAIN_HEAD_DIM * 2,
        0,
    )

    swa_bt = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)
    swa_meta = _make_attn_meta(
        block_tables=swa_bt,
        seq_lens=torch.tensor([8, 11], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    ctx_kwargs = {
        "deepseek_v41_window_starts": torch.tensor([3, 0], dtype=torch.int32),
        "deepseek_v41_replay_starts": torch.tensor([5, 4], dtype=torch.int32),
    }

    getitem_calls = 0
    orig_getitem = PagedCacheRows.__getitem__

    def _spy_getitem(
        self: PagedCacheRows, idx: tuple[torch.Tensor, torch.Tensor]
    ) -> Any:
        nonlocal getitem_calls
        getitem_calls += 1
        return orig_getitem(self, idx)

    monkeypatch.setattr(PagedCacheRows, "__getitem__", _spy_getitem)

    # Mode 1: Single-bank BF16 PagedCacheRows (null_block=True) -> gather_paged_windows.
    layer.swa_cache_layer.bind_kv_cache(torch.empty(0), raw_cache=raw_3d, layout=layout)
    assert layer.swa_cache_layer.paged_cache_rows is not None
    assert layer.swa_cache_layer.paged_cache_rows.null_block is True
    assert layer.swa_cache_layer.paged_cache_rows.supports_paged_windows is True

    out_paged = torch.empty_like(q)
    layer.ced_decoder = False
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: swa_meta}, ctx_kwargs)
    ):
        layer.forward_mqa(q, kv, pos, out_paged)
    assert getitem_calls == 0
    assert len(stub_dsv41_ops["gather_swa_windows"]) == 1
    assert stub_dsv41_ops["gather_swa_windows"][0]["pool"] is raw_3d
    swa_kv_paged = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"]
    min_pos_enc = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_min_pos"]
    assert min_pos_enc.tolist() == [3, 0]
    assert torch.isfinite(swa_kv_paged).all()
    assert torch.isfinite(out_paged).all()
    assert int(swa_kv_paged[6].abs().max().item()) == 0
    assert int(swa_kv_paged[11].abs().max().item()) == 0
    assert torch.equal(swa_kv_paged[7:11], logical_bf16[3])

    seq0_hist = torch.cat(
        (
            torch.zeros((3, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            logical_bf16[1, 3:4],
            logical_bf16[2, 0:2],
            kv[:2],
        ),
        dim=0,
    )
    seq1_hist = torch.cat(
        (
            torch.zeros((4, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            logical_bf16[3],
            torch.zeros((1, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            kv[2:4],
        ),
        dim=0,
    )
    ref_null = _reference_attention_fp32(
        q,
        [seq0_hist, seq1_hist],
        None,
        None,
        layer.attn_sink,
        [8, 11],
        [0, 2, 4],
        sm_scale=layer.scale,
        window_size=layer.window_size,
        min_pos_by_seq=[3, 0],
    )
    assert torch.allclose(out_paged.float(), ref_null.float(), atol=2e-2, rtol=2e-2)

    # Check ced_decoder=True uses max(window_starts, replay_starts) = [5, 4].
    layer.ced_decoder = True
    out_dec = torch.empty_like(q)
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: swa_meta}, ctx_kwargs)
    ):
        layer.forward_mqa(q, kv, pos, out_dec)
    min_pos_dec = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_min_pos"]
    swa_kv_dec = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"]
    assert min_pos_dec.tolist() == [5, 4]
    assert int(swa_kv_dec[:5].abs().max().item()) == 0
    layer.ced_decoder = False

    # Mode 2: Unbound eager Tensor SWA cache -> flat indexing bit-exact parity.
    unbound_swa = logical_bf16.clone()
    unbound_swa[0].zero_()
    layer.swa_cache_layer.kv_cache = unbound_swa
    out_unbound = torch.empty_like(q)
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: swa_meta}, ctx_kwargs)
    ):
        layer.forward_mqa(q, kv, pos, out_unbound)
    swa_kv_unbound = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"]
    assert torch.equal(
        swa_kv_paged.view(torch.uint16), swa_kv_unbound.view(torch.uint16)
    )
    assert torch.equal(out_paged.view(torch.uint16), out_unbound.view(torch.uint16))

    # Zero-column block table and window_size=0 yield swa_kv=None safely.
    zero_col_meta = _make_attn_meta(
        block_tables=torch.empty((2, 0), dtype=torch.int32),
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: zero_col_meta})
    ):
        layer.forward_mqa(q, kv, pos, torch.empty_like(q))
    assert stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"] is None

    layer.window_size = 0
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: swa_meta})
    ):
        layer.forward_mqa(q, kv, pos, torch.empty_like(q))
    assert stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"] is None


def test_should_pass_3d_pools_and_preserve_unaligned_sparse_gather_parity(
    cpu_dist_ctx: None,
    stub_dsv41_ops: dict[str, list[Any]],
) -> None:
    del cpu_dist_ctx
    vcfg = _make_dsv41_vllm_config(
        block_size=16, sliding_window=4, index_topk=4, max_model_len=64
    )
    topk_buf = torch.tensor(
        [[0, 2, 5, 7], [1, 3, -1, 15], [0, 4, 5, -1], [0, 1, 2, 3]],
        dtype=torch.int32,
    )
    with set_current_vllm_config(vcfg):
        layer2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_buf)
        layer20 = DeepseekV41Attention(vcfg, "model.layers.20.attn", topk_buf)
    _init_attention_weights(layer2, seed=21)
    _init_attention_weights(layer20, seed=22)

    q = torch.randn((4, NUM_HEADS, MAIN_HEAD_DIM), dtype=torch.bfloat16) * 0.1
    kv = torch.randn((4, MAIN_HEAD_DIM), dtype=torch.bfloat16) * 0.1
    pos = torch.tensor([1, 3, 5, 0], dtype=torch.int32)
    swa_dummy = torch.zeros((4, 4, MAIN_HEAD_DIM), dtype=torch.bfloat16)
    layer2.swa_cache_layer.kv_cache = swa_dummy
    layer20.swa_cache_layer.kv_cache = swa_dummy
    swa_meta = _make_attn_meta(
        block_tables=torch.tensor([[1, 2], [2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([8, 6], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
    )

    def _fill_fp8_cache(
        pcr: PagedCacheRows, block_table: torch.Tensor, page_size: int
    ) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
        ground_truth: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        by_bid: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        gen = torch.Generator().manual_seed(303)
        for s_idx in range(block_table.shape[0]):
            for col in range(block_table.shape[1]):
                bid = int(block_table[s_idx, col].item())
                if bid < (1 if pcr.null_block else 0):
                    continue
                if bid not in by_bid:
                    latents = torch.randn(
                        (page_size, MAIN_HEAD_DIM), generator=gen, dtype=torch.bfloat16
                    )
                    slots = torch.arange(page_size, dtype=torch.int32) + bid * page_size
                    pcr.write(slots, latents)
                    by_bid[bid] = cache_mod.cache_op.dsv41_pack_fp8_kv_rows(latents)
                nope_p, rope_p = by_bid[bid]
                for off in range(page_size):
                    ground_truth[(s_idx, col * page_size + off)] = (
                        nope_p[off].clone(),
                        rope_p[off].clone(),
                    )
        return ground_truth

    def _assert_gather_matches(
        ground_truth: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
        cu_q_lens: torch.Tensor,
        topk: torch.Tensor,
        gathered_nope: torch.Tensor,
        gathered_rope: torch.Tensor,
    ) -> None:
        num_seqs = cu_q_lens.shape[0] - 1
        for t in range(topk.shape[0]):
            seq_id = next(
                (s for s in range(num_seqs) if cu_q_lens[s] <= t < cu_q_lens[s + 1]),
                None,
            )
            for k in range(topk.shape[1]):
                idx = int(topk[t, k].item())
                q_row = k // 4
                q_off = (k % 4) * ROPE_WORDS
                actual_nope = gathered_nope[t, k]
                actual_rope = gathered_rope[t, q_row, q_off : q_off + ROPE_WORDS]
                if seq_id is not None and (seq_id, idx) in ground_truth:
                    exp_nope, exp_rope = ground_truth[(seq_id, idx)]
                    assert torch.equal(actual_nope, exp_nope)
                    assert torch.equal(actual_rope, exp_rope)
                else:
                    assert int(actual_nope.abs().max().item()) == 0
                    assert int(actual_rope.abs().max().item()) == 0

    # Case 1: Unit-scale 3D root pool (L20, page_scale=1) -> exact object identity.
    bt_unit = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    raw_unit = torch.zeros((4, 8, ROW_WORDS), dtype=torch.int32)
    rope_unit = torch.zeros((4, 2, ROW_WORDS), dtype=torch.int32)
    layer20.bind_kv_cache(
        (raw_unit, rope_unit),
        raw_cache=raw_unit,
        layout=((4, 8, MAIN_HEAD_DIM), torch.uint8, 8 * ROW_BYTES, 0),
        rope_pool=rope_unit,
    )
    assert layer20.paged_cache_rows is not None
    gt_unit = _fill_fp8_cache(layer20.paged_cache_rows, bt_unit, page_size=8)
    swa_seq_lens = cast(torch.Tensor, swa_meta.seq_lens)
    swa_cu_q = cast(torch.Tensor, swa_meta.query_start_loc)
    meta_unit = _make_attn_meta(bt_unit, swa_seq_lens, swa_cu_q)
    with override_forward_context(
        _forward_ctx(
            {layer20.swa_cache_layer.prefix: swa_meta, layer20.prefix: meta_unit}
        )
    ):
        layer20.forward_mqa(q, kv, pos, torch.empty_like(q))
    call_unit = stub_dsv41_ops["gather_sparse_fp8_kv"][-1]
    attn_unit = stub_dsv41_ops["sparse_mla_swa_attention"][-1]
    assert call_unit["kv_pool"] is raw_unit
    assert call_unit["rope_pool"] is rope_unit
    _assert_gather_matches(
        gt_unit,
        swa_cu_q,
        topk_buf,
        attn_unit["gathered_nope"],
        attn_unit["gathered_rope"],
    )

    # Case 2: Paired homogeneous pool (L2, page_scale=2) -> contiguous 3D offset==0.
    raw_pair = torch.zeros((4, 16, ROW_WORDS), dtype=torch.int32)
    rope_pair = torch.zeros((4, 4, ROW_WORDS), dtype=torch.int32)
    layer2.bind_kv_cache(
        (raw_pair, rope_pair),
        raw_cache=raw_pair,
        layout=((4, 8, MAIN_HEAD_DIM), torch.uint8, 16 * ROW_BYTES, 8 * ROW_BYTES),
        rope_pool=rope_pair,
    )
    assert layer2.paged_cache_rows is not None
    gt_pair = _fill_fp8_cache(layer2.paged_cache_rows, bt_unit, page_size=8)
    with override_forward_context(
        _forward_ctx(
            {layer2.swa_cache_layer.prefix: swa_meta, layer2.prefix: meta_unit}
        )
    ):
        layer2.forward_mqa(q, kv, pos, torch.empty_like(q))
    call_pair = stub_dsv41_ops["gather_sparse_fp8_kv"][-1]
    attn_pair = stub_dsv41_ops["sparse_mla_swa_attention"][-1]
    assert call_pair["kv_pool"].is_contiguous()
    assert call_pair["kv_pool"].storage_offset() == 0
    assert call_pair["rope_pool"].storage_offset() == 0
    assert call_pair["kv_pool"].shape == (8, 8, ROW_WORDS)
    assert call_pair["rope_pool"].shape == (8, 2, ROW_WORDS)
    _assert_gather_matches(
        gt_pair,
        swa_cu_q,
        topk_buf,
        attn_pair["gathered_nope"],
        attn_pair["gathered_rope"],
    )

    # Case 3: Single-bank unaligned pages(...) is None with poisoned null block.
    bt_null = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)
    unaligned_cases = [
        (
            torch.zeros((44, ROW_WORDS), dtype=torch.int32),
            torch.zeros((11, ROW_WORDS), dtype=torch.int32),
            ((4, 4, MAIN_HEAD_DIM), torch.uint8, 10 * ROW_BYTES, 2 * ROW_BYTES),
            4,
            bt_null,
        ),
        (
            torch.zeros((8, ROW_WORDS), dtype=torch.int32),
            torch.zeros((2, ROW_WORDS), dtype=torch.int32),
            ((4, 2, MAIN_HEAD_DIM), torch.uint8, 2 * ROW_BYTES, 0),
            2,
            bt_null,
        ),
        (
            torch.zeros((20, ROW_WORDS), dtype=torch.int32)[4:],
            torch.zeros((5, ROW_WORDS), dtype=torch.int32)[1:],
            ((4, 4, MAIN_HEAD_DIM), torch.uint8, 4 * ROW_BYTES, 0),
            4,
            bt_null,
        ),
    ]
    for raw_c, rope_c, lay_c, ps_c, bt_c in unaligned_cases:
        layer2.bind_kv_cache(
            (raw_c, rope_c),
            raw_cache=raw_c,
            layout=lay_c,
            rope_pool=rope_c,
        )
        pcr_u = layer2.paged_cache_rows
        assert pcr_u is not None and pcr_u.rope is not None and pcr_u.null_block is True
        gt_u = _fill_fp8_cache(pcr_u, bt_c, page_size=ps_c)
        null_phys = pcr_u.row_indices(torch.arange(ps_c, dtype=torch.int32)).tolist()
        for pr in null_phys:
            pcr_u.raw.reshape(-1, ROW_WORDS)[pr].fill_(0x3F803F80)
            q_off = (pr % 4) * ROPE_WORDS
            pcr_u.rope.reshape(-1, ROW_WORDS)[
                pr // 4, q_off : q_off + ROPE_WORDS
            ].fill_(0x3F803F80)
        meta_c = _make_attn_meta(bt_c, swa_seq_lens, swa_cu_q)
        with override_forward_context(
            _forward_ctx(
                {layer2.swa_cache_layer.prefix: swa_meta, layer2.prefix: meta_c}
            )
        ):
            layer2.forward_mqa(q, kv, pos, torch.empty_like(q))
        call_u = stub_dsv41_ops["gather_sparse_fp8_kv"][-1]
        attn_u = stub_dsv41_ops["sparse_mla_swa_attention"][-1]
        assert call_u["kv_pool"].ndim == 3
        assert call_u["kv_pool"].is_contiguous()
        assert call_u["kv_pool"].storage_offset() == 0
        assert call_u["rope_pool"].storage_offset() == 0
        assert call_u["page_size"] % 4 == 0
        assert torch.equal(attn_u["topk_indices"], topk_buf)
        _assert_gather_matches(
            gt_u,
            swa_cu_q,
            topk_buf,
            attn_u["gathered_nope"],
            attn_u["gathered_rope"],
        )

    # Case 4: Unbound (kv_pool, rope_pool) tuple and None fallback.
    layer20.kv_cache = (raw_unit, rope_unit)
    with override_forward_context(
        _forward_ctx(
            {layer20.swa_cache_layer.prefix: swa_meta, layer20.prefix: meta_unit}
        )
    ):
        layer20.forward_mqa(q, kv, pos, torch.empty_like(q))
    assert stub_dsv41_ops["gather_sparse_fp8_kv"][-1]["kv_pool"] is raw_unit

    layer20.kv_cache = cast(Any, None)
    calls_before_none = len(stub_dsv41_ops["gather_sparse_fp8_kv"])
    with override_forward_context(
        _forward_ctx(
            {layer20.swa_cache_layer.prefix: swa_meta, layer20.prefix: meta_unit}
        )
    ):
        layer20.forward_mqa(q, kv, pos, torch.empty_like(q))
    assert len(stub_dsv41_ops["gather_sparse_fp8_kv"]) == calls_before_none
    with override_forward_context(_forward_ctx(None)):
        layer20.forward_kv(pos, torch.zeros((4, HIDDEN_SIZE), dtype=torch.bfloat16))


def test_should_match_fp32_reference_across_prefill_decode_and_regimes(
    cpu_dist_ctx: None,
    stub_dsv41_ops: dict[str, list[Any]],
) -> None:
    del cpu_dist_ctx
    vcfg = _make_dsv41_vllm_config(
        block_size=16,
        sliding_window=4,
        index_topk=4,
        candidate_source_layer_id=19,
        candidate_block_size=4,
        candidate_topk_blocks=4,
        max_model_len=128,
        max_position_embeddings=128,
        swa_bounded_replay=True,
    )
    topk_buf = torch.full((16, 4), -1, dtype=torch.int32)
    cand_buf = torch.full((16, 4), -1, dtype=torch.int32)
    with set_current_vllm_config(vcfg):
        l0 = DeepseekV41Attention(vcfg, "model.layers.0.attn", topk_buf)
        l2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_buf, None, cand_buf)
        l3 = DeepseekV41Attention(vcfg, "model.layers.3.attn", topk_buf, None, cand_buf)
        l8 = DeepseekV41Attention(vcfg, "model.layers.8.attn", topk_buf, None, cand_buf)
        l14 = DeepseekV41Attention(
            vcfg, "model.layers.14.attn", topk_buf, None, cand_buf
        )
        l19 = DeepseekV41Attention(
            vcfg, "model.layers.19.attn", topk_buf, None, cand_buf
        )
        l20 = DeepseekV41Attention(
            vcfg, "model.layers.20.attn", topk_buf, None, cand_buf
        )

    all_layers = (l0, l2, l3, l8, l14, l19, l20)
    src_by_layer = {
        l0: None,
        l2: l2,
        l3: l2,
        l8: l8,
        l14: l14,
        l19: l14,
        l20: l20,
    }
    for seed, layer in enumerate(all_layers, start=41):
        _init_attention_weights(layer, seed=seed, fp8_wo_a=True)
        layer.process_weights_after_loading(torch.bfloat16)

    runner = _RunnerTestHarness(vcfg, torch.device("cpu"), max_num_reqs=4)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 24)
    mgr.initialize_kv_cache(kv_cfg)

    bt_by_layer: dict[str, torch.Tensor] = {}
    next_bid = 1
    for group in kv_cfg.kv_cache_groups:
        is_state = group.kv_cache_spec.block_size == 1
        cols = 1 if is_state else 2
        g_bt = torch.arange(next_bid, next_bid + 2 * cols, dtype=torch.int32).reshape(
            2, cols
        )
        next_bid += 2 * cols
        for layer_name in group.layer_names:
            bt_by_layer[layer_name] = g_bt

    def _build_meta_map(
        seq_lens: list[int], qsl: list[int]
    ) -> dict[str, AttentionMetadata]:
        sl_t = torch.tensor(seq_lens, dtype=torch.int32)
        qsl_t = torch.tensor(qsl, dtype=torch.int32)
        meta_map: dict[str, AttentionMetadata] = {
            layer_name: _make_attn_meta(g_bt, sl_t, qsl_t)
            for layer_name, g_bt in bt_by_layer.items()
        }
        for lyr in all_layers:
            if lyr.prefix not in meta_map and lyr.compressed_cache_prefix in meta_map:
                meta_map[lyr.prefix] = meta_map[lyr.compressed_cache_prefix]
        return meta_map

    def _assert_layer_fp32_parity(
        lyr: DeepseekV41Attention,
        src_lyr: DeepseekV41Attention | None,
        actual: torch.Tensor,
        h_q: torch.Tensor,
        pos_q: torch.Tensor,
        swa_by_seq: list[torch.Tensor],
        attn_rec: dict[str, Any],
        seq_lens: list[int],
        cu_q_lens: list[int],
        min_pos_by_seq: list[int] | None,
    ) -> None:
        num_active = cu_q_lens[-1]
        assert torch.isfinite(actual).all()
        qr_kv, _ = lyr.fused_wqa_wkv(h_q)
        qr = lyr.q_norm(qr_kv[:, :Q_LORA_RANK])
        q_rot = lyr.rotate(lyr.wq_b(qr).view(-1, NUM_HEADS, MAIN_HEAD_DIM), pos_q)
        comp_by_seq: list[torch.Tensor] | None = None
        topk_t: torch.Tensor | None = None
        ratio = lyr.compress_ratio
        if ratio > 0 and src_lyr is not None:
            topk_t = attn_rec["topk_indices"]
            pcr = src_lyr.paged_cache_rows
            assert pcr is not None and pcr.rope is not None
            bt_seq = bt_by_layer[src_lyr.prefix].tolist()
            comp_by_seq = []
            for s_idx, s_len in enumerate(seq_lens):
                n_comp = s_len // ratio
                rows_f32: list[torch.Tensor] = []
                for j in range(n_comp):
                    bid = bt_seq[s_idx][j // pcr.shape[1]]
                    slot = bid * pcr.shape[1] + (j % pcr.shape[1])
                    phys = int(pcr.row_indices(torch.tensor([slot])).item())
                    nope_w = pcr.raw.reshape(-1, ROW_WORDS)[phys]
                    rope_q = pcr.rope.reshape(-1, ROPE_WORDS)[phys]
                    rows_f32.append(_dequantize_packed_fp8_row(nope_w, rope_q))
                comp_by_seq.append(torch.stack(rows_f32, dim=0))

        ref_mqa = _reference_attention_fp32(
            q_rot[:num_active],
            swa_by_seq,
            comp_by_seq,
            None if topk_t is None else topk_t[:num_active],
            lyr.attn_sink,
            seq_lens,
            cu_q_lens,
            sm_scale=lyr.scale,
            window_size=lyr.window_size,
            compress_ratio=max(1, ratio),
            min_pos_by_seq=min_pos_by_seq,
        )
        ref_z = _cpu_inverse_rope_wo_a(
            ref_mqa.bfloat16(),
            pos_q[:num_active],
            lyr.rotary_emb.cos_sin_cache,
            cast(torch.Tensor, lyr.wo_a.weight),
            cast(torch.Tensor, lyr.wo_a.weight_scale),
        )
        ref_out = lyr.wo_b(ref_z)
        assert torch.allclose(
            actual[:num_active].float(), ref_out.float(), atol=5e-2, rtol=5e-2
        )

    # Step 1 (Cold Prefill): seq 0 len=6, seq 1 len=4, padded to 12 tokens.
    gen = torch.Generator().manual_seed(909)
    h_prefill = torch.randn((12, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
    pos_prefill = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 0], dtype=torch.int32)
    meta_prefill = _build_meta_map([6, 4], [0, 6, 10])

    prefill_outs: dict[DeepseekV41Attention, tuple[torch.Tensor, dict[str, Any]]] = {}
    with override_forward_context(_forward_ctx(meta_prefill)):
        for lyr in all_layers:
            out_p = lyr(pos_prefill, h_prefill)
            prefill_outs[lyr] = (
                out_p,
                stub_dsv41_ops["sparse_mla_swa_attention"][-1],
            )
            if lyr is l19:
                assert int(cand_buf[5, 0].item()) >= 0

    for lyr in all_layers:
        out_p, attn_p = prefill_outs[lyr]
        swa_hist_p = lyr.rotate(lyr.project_context_kv(h_prefill), pos_prefill)
        _assert_layer_fp32_parity(
            lyr,
            src_by_layer[lyr],
            out_p,
            h_prefill,
            pos_prefill,
            [swa_hist_p[:6], swa_hist_p[6:10]],
            attn_p,
            [6, 4],
            [0, 6, 10],
            None,
        )

    # Step 2 (Warm Decode / MTP): seq 0 +2 [6, 7], seq 1 +1 [4], replay_starts=[2, 1].
    h_decode = torch.randn((4, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
    pos_decode = torch.tensor([6, 7, 4, 0], dtype=torch.int32)
    meta_decode = _build_meta_map([8, 5], [0, 2, 3])
    dec_kwargs = {"deepseek_v41_window_starts": torch.tensor([2, 1], dtype=torch.int32)}

    # Exercise ced_global_from_encoder=True on L20 via write_encoder_global_kv,
    # verifying both compressed KV and indexer K cache writes (via public helper).
    assert l20.indexer is not None
    l20_k_rows = l20.indexer.k_cache.paged_cache_rows
    assert isinstance(l20_k_rows, PagedCacheRows)
    idx_k_before = l20_k_rows.raw.clone()
    l20.ced_global_from_encoder = True
    decode_outs: dict[DeepseekV41Attention, tuple[torch.Tensor, dict[str, Any]]] = {}
    with override_forward_context(_forward_ctx(meta_decode, dec_kwargs)):
        l20.write_encoder_global_kv(h_decode, pos_decode)
        assert not torch.equal(l20_k_rows.raw, idx_k_before)
        for lyr in all_layers:
            out_d = lyr(pos_decode, h_decode)
            decode_outs[lyr] = (
                out_d,
                stub_dsv41_ops["sparse_mla_swa_attention"][-1],
            )

    for lyr in all_layers:
        out_d, attn_d = decode_outs[lyr]
        swa_hist_p = lyr.rotate(lyr.project_context_kv(h_prefill), pos_prefill)
        swa_hist_d = lyr.rotate(lyr.project_context_kv(h_decode), pos_decode)
        _assert_layer_fp32_parity(
            lyr,
            src_by_layer[lyr],
            out_d,
            h_decode,
            pos_decode,
            [
                torch.cat((swa_hist_p[:6], swa_hist_d[:2]), dim=0),
                torch.cat((swa_hist_p[6:10], swa_hist_d[2:3]), dim=0),
            ],
            attn_d,
            [8, 5],
            [0, 2, 3],
            [2, 1],
        )

    # Verify forward_kv writes caches cleanly on L0 (no compressor) and L2 (indexer K).
    assert l2.indexer is not None
    l2_k_rows = l2.indexer.k_cache.paged_cache_rows
    assert isinstance(l2_k_rows, PagedCacheRows)
    l2_k_before = l2_k_rows.raw.clone()
    h_fkv = torch.randn((4, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16) * 2.0
    with override_forward_context(_forward_ctx(meta_decode, dec_kwargs)):
        l0.forward_kv(pos_decode, h_fkv)
        l2.forward_kv(pos_decode, h_fkv)
    assert not torch.equal(l2_k_rows.raw, l2_k_before)
    assert prefill_outs[l2][1]["topk_indices"].shape == (12, 4)
    assert prefill_outs[l20][1]["topk_indices"].shape == (12, 4)
    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].abs().max().item()) == 0


def test_should_handle_dspark_non_causal_splicing_and_multi_rank_tp_sp_sharding(
    cpu_dist_ctx: None,
    stub_dsv41_ops: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    vcfg = _make_dsv41_vllm_config(
        block_size=4, sliding_window=4, index_topk=4, max_model_len=64
    )
    topk_buf = torch.full((8, 4), -1, dtype=torch.int32)
    with set_current_vllm_config(vcfg):
        draft = DeepseekV41Attention(vcfg, "model.layers.0.attn", topk_buf)
        src_l2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_buf)
    _init_attention_weights(draft, seed=61, fp8_wo_a=True)
    draft.process_weights_after_loading(torch.bfloat16)
    draft.non_causal_multi_token_decode = True
    draft.dspark_block_size = 2

    raw_swa = torch.zeros((4, 4 * 2, ROW_WORDS), dtype=torch.int32)
    draft.swa_cache_layer.bind_kv_cache(
        torch.empty(0),
        raw_cache=raw_swa,
        layout=((4, 4, MAIN_HEAD_DIM), torch.bfloat16, 4 * MAIN_HEAD_DIM * 2, 0),
    )
    assert draft.swa_cache_layer.paged_cache_rows is not None
    gen = torch.Generator().manual_seed(62)
    prefix_kv = torch.randn((3, MAIN_HEAD_DIM), generator=gen, dtype=torch.bfloat16)
    draft.swa_cache_layer.paged_cache_rows.write(
        torch.tensor([4, 5, 6], dtype=torch.int32), prefix_kv
    )
    before_raw = raw_swa.clone()

    h_draft = torch.randn((4, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
    pos_draft = torch.tensor([3, 4, 0, 0], dtype=torch.int32)
    live_kv = draft.rotate(draft.project_context_kv(h_draft), pos_draft)
    qr_kv, _ = draft.fused_wqa_wkv(h_draft)
    q_rot = draft.rotate(
        draft.wq_b(draft.q_norm(qr_kv[:, :Q_LORA_RANK])).view(
            -1, NUM_HEADS, MAIN_HEAD_DIM
        ),
        pos_draft,
    )
    for dspark_bs in (2, 1):
        draft.dspark_block_size = dspark_bs
        meta_draft = _make_attn_meta(
            block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([3 + dspark_bs], dtype=torch.int32),
            query_start_loc=torch.tensor([0, dspark_bs], dtype=torch.int32),
        )
        with override_forward_context(
            _forward_ctx(
                {draft.swa_cache_layer.prefix: meta_draft},
                {"deepseek_v41_window_starts": torch.tensor([1], dtype=torch.int32)},
            )
        ):
            out_draft = draft(pos_draft, h_draft)

        assert torch.equal(raw_swa, before_raw)
        full_hist = torch.cat((prefix_kv, live_kv[:dspark_bs]), dim=0)
        ref_nc = _reference_attention_fp32(
            q_rot[:dspark_bs],
            [full_hist],
            None,
            None,
            draft.attn_sink,
            [3 + dspark_bs],
            [0, dspark_bs],
            sm_scale=draft.scale,
            window_size=draft.window_size + dspark_bs,
            min_pos_by_seq=[1],
            non_causal=True,
        )
        ref_out = draft.wo_b(
            _cpu_inverse_rope_wo_a(
                ref_nc.bfloat16(),
                pos_draft[:dspark_bs],
                draft.rotary_emb.cos_sin_cache,
                cast(torch.Tensor, draft.wo_a.weight),
                cast(torch.Tensor, draft.wo_a.weight_scale),
            )
        )
        assert torch.allclose(
            out_draft[:dspark_bs].float(), ref_out.float(), atol=5e-2, rtol=5e-2
        )

    for dspark_bs, read_non_causal in ((0, False), (2, True)):
        raw_cap = torch.zeros((5, 4 * 2, ROW_WORDS), dtype=torch.int32)
        draft.swa_cache_layer.bind_kv_cache(
            torch.empty(0),
            raw_cache=raw_cap,
            layout=((5, 4, MAIN_HEAD_DIM), torch.bfloat16, 4 * MAIN_HEAD_DIM * 2, 0),
        )
        draft.non_causal_multi_token_decode = False
        draft.dspark_block_size = dspark_bs
        h_p8 = torch.randn((8, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
        pos_p8 = torch.arange(8, dtype=torch.int32)
        bt_cap = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        meta_p8 = _make_attn_meta(
            bt_cap,
            torch.tensor([8], dtype=torch.int32),
            torch.tensor([0, 8], dtype=torch.int32),
        )
        with override_forward_context(
            _forward_ctx({draft.swa_cache_layer.prefix: meta_p8})
        ):
            _ = draft(pos_p8, h_p8)

        draft.non_causal_multi_token_decode = read_non_causal
        h_d1 = torch.randn((1, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
        pos_d1 = torch.tensor([8], dtype=torch.int32)
        meta_d1 = _make_attn_meta(
            bt_cap,
            torch.tensor([9], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        )
        with override_forward_context(
            _forward_ctx({draft.swa_cache_layer.prefix: meta_d1})
        ):
            out_d1 = draft(pos_d1, h_d1)

        kv_p8 = draft.rotate(draft.project_context_kv(h_p8), pos_p8)
        kv_d1 = draft.rotate(draft.project_context_kv(h_d1), pos_d1)
        qr_d1, _ = draft.fused_wqa_wkv(h_d1)
        q_d1 = draft.rotate(
            draft.wq_b(draft.q_norm(qr_d1[:, :Q_LORA_RANK])).view(
                -1, NUM_HEADS, MAIN_HEAD_DIM
            ),
            pos_d1,
        )
        ref_cap = _reference_attention_fp32(
            q_d1,
            [torch.cat((kv_p8, kv_d1), dim=0)],
            None,
            None,
            draft.attn_sink,
            [9],
            [0, 1],
            sm_scale=draft.scale,
            window_size=draft.window_size + (dspark_bs if read_non_causal else 0),
            non_causal=read_non_causal,
        )
        ref_d1 = draft.wo_b(
            _cpu_inverse_rope_wo_a(
                ref_cap.bfloat16(),
                pos_d1,
                draft.rotary_emb.cos_sin_cache,
                cast(torch.Tensor, draft.wo_a.weight),
                cast(torch.Tensor, draft.wo_a.weight_scale),
            )
        )
        assert torch.allclose(out_d1.float(), ref_d1.float(), atol=5e-2, rtol=5e-2)
        if not read_non_causal:
            swa_kv_cap = stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"]
            assert torch.equal(swa_kv_cap[0], kv_p8[4])

    draft.window_size = 0
    draft.dspark_block_size = 0
    with override_forward_context(
        _forward_ctx({draft.swa_cache_layer.prefix: meta_draft})
    ):
        draft.forward_mqa(q_rot, live_kv, pos_draft, torch.empty_like(q_rot))
    assert stub_dsv41_ops["sparse_mla_swa_attention"][-1]["swa_kv"] is None

    # write_encoder_global_kv: no compressor, sp_sharded=True, compressor-only layer.
    draft.write_encoder_global_kv(h_draft, pos_draft)
    gathered_calls = 0

    def _mock_sp_all_gather(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal gathered_calls
        gathered_calls += 1
        return torch.cat((tensor, tensor), dim=0)

    monkeypatch.setattr(attn_mod, "sp_all_gather", _mock_sp_all_gather)
    _init_attention_weights(src_l2, seed=63)
    with override_forward_context(_forward_ctx({})):
        src_l2.write_encoder_global_kv(h_draft[:2], pos_draft[:3], sp_sharded=True)
    assert gathered_calls == 1

    vcfg_comp_only = _make_dsv41_vllm_config(
        compress_ratios=[0, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[1],
        max_model_len=64,
    )
    with set_current_vllm_config(vcfg_comp_only):
        comp_only = DeepseekV41Attention(vcfg_comp_only, "model.layers.2.attn")
    _init_attention_weights(comp_only, seed=64)
    assert comp_only.compressor is not None and comp_only.indexer is None
    with override_forward_context(_forward_ctx({})):
        comp_only.write_encoder_global_kv(h_draft, pos_draft)

    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    vcfg_tp2 = _make_dsv41_vllm_config(
        num_attention_heads=16, o_groups=2, tp_size=2, max_model_len=64
    )
    with set_current_vllm_config(vcfg_tp2):
        tp_layer = DeepseekV41Attention(vcfg_tp2, "model.layers.0.attn")
    assert tp_layer.n_local_heads == 8
    assert tp_layer.n_local_groups == 1
    assert tp_layer.wo_a.bmm_batch_size == 1


def test_should_write_rotated_non_replayed_indexer_k_on_kv_only_paths(
    cpu_dist_ctx: None,
    stub_dsv41_ops: dict[str, list[Any]],
) -> None:
    del cpu_dist_ctx, stub_dsv41_ops
    vcfg = _make_dsv41_vllm_config(
        block_size=8,
        sliding_window=2,
        index_n_heads=4,
        index_topk=4,
        max_model_len=64,
        max_position_embeddings=64,
        max_num_seqs=2,
    )
    topk_buf = torch.full((8, 4), -1, dtype=torch.int32)
    cand_buf = torch.full((8, 4), -1, dtype=torch.int32)
    with set_current_vllm_config(vcfg):
        l2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_buf, None, cand_buf)
    _init_attention_weights(l2, seed=91, fp8_wo_a=True)
    l2.process_weights_after_loading(torch.bfloat16)
    assert l2.compressor is not None and l2.indexer is not None

    dev = torch.device("cpu")
    runner = _RunnerTestHarness(vcfg, dev, max_num_reqs=2, max_model_len=64)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 12)
    mgr.initialize_kv_cache(kv_cfg)

    k_pcr = l2.indexer.k_cache.paged_cache_rows
    assert isinstance(k_pcr, PagedCacheRows)
    sentinel_bf16 = torch.full((INDEX_HEAD_DIM,), 7.5, dtype=torch.bfloat16)
    sentinel_u16 = sentinel_bf16.view(torch.uint16).to(torch.int64)
    sentinel_word = int(sentinel_u16[0].item()) | (int(sentinel_u16[1].item()) << 16)
    sentinel_i32 = int(
        torch.tensor(sentinel_word, dtype=torch.uint32).view(torch.int32).item()
    )
    k_pcr.raw.fill_(sentinel_i32)

    def _read_k_slot(block_id: int, row_id: int) -> torch.Tensor:
        slot = torch.tensor([block_id * k_pcr.shape[1] + row_id], dtype=torch.int32)
        rec_idx = int(k_pcr.row_indices(slot).item())
        words_2d = k_pcr.raw.reshape(-1, INDEX_HEAD_DIM // 2)
        words = words_2d[rec_idx]
        even = (words << BF16_SHIFT_BITS).view(torch.float32).to(torch.bfloat16)
        odd = (words & BF16_ODD_MASK).view(torch.float32).to(torch.bfloat16)
        return torch.stack((even, odd), dim=-1).flatten(-2)

    gen = torch.Generator().manual_seed(92)
    pos = torch.arange(6, dtype=torch.int32)
    hidden = (torch.randn((6, HIDDEN_SIZE), generator=gen) * 0.4).bfloat16()
    hidden[0] += 0.5
    hidden[2] -= 0.5
    hidden[4] += 0.8

    meta = _make_attn_meta(
        block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        positions=pos,
    )
    meta_dict = {
        l2.swa_cache_layer.prefix: meta,
        l2.prefix: meta,
        l2.indexer.k_cache.prefix: meta,
    }
    ctx_replay = _forward_ctx(
        meta_dict,
        additional_kwargs={
            "deepseek_v41_window_starts": torch.tensor([2], dtype=torch.int32)
        },
    )
    with override_forward_context(ctx_replay):
        l2.forward_kv(pos, hidden)
        latent = l2.compressor(l2.compressor.project(hidden), pos)
    k_pre, _ = l2.indexer.wk(latent)
    k_normed = l2.indexer.k_norm(k_pre)
    rope_cs = cast(torch.Tensor, l2.indexer_rotary_emb.cos_sin_cache)
    expected_group_rope = _cpu_rope(k_normed, (pos // 2) * 2, rope_cs)
    unrounded_rope = _cpu_rope(k_normed, pos, rope_cs)

    # Group 0 (pos=1, outside replay [2,4)): exact boundary & group-aligned RoPE.
    slot0 = _read_k_slot(1, 0)
    assert torch.equal(slot0, expected_group_rope[1])
    assert not torch.equal(slot0, expected_group_rope[0])
    assert not torch.equal(slot0, unrounded_rope[1])
    # Group 1 (pos=3, inside replay [2,4)): skipped -> sentinel preserved.
    assert torch.equal(_read_k_slot(1, 1), sentinel_bf16)
    # Group 2 (pos=5, outside replay [2,4)): exact boundary & group-aligned RoPE.
    slot2 = _read_k_slot(1, 2)
    assert torch.equal(slot2, expected_group_rope[5])
    assert not torch.equal(slot2, expected_group_rope[4])
    assert not torch.equal(slot2, unrounded_rope[5])
    assert torch.equal(_read_k_slot(1, 3), sentinel_bf16)

    # Encoder global-KV path without replay writes both compressed groups to block 3.
    pos_enc = torch.arange(4, dtype=torch.int32)
    h_enc = (torch.randn((4, HIDDEN_SIZE), generator=gen) * 0.4).bfloat16()
    meta_enc = _make_attn_meta(
        block_tables=torch.tensor([[3, 4]], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        positions=pos_enc,
    )
    ctx_enc = _forward_ctx(
        {
            l2.prefix: meta_enc,
            l2.indexer.k_cache.prefix: meta_enc,
        }
    )
    with override_forward_context(ctx_enc):
        l2.write_encoder_global_kv(h_enc, pos_enc, sp_sharded=False)
        latent_enc = l2.compressor(l2.compressor.project(h_enc), pos_enc)
    k_pre_enc, _ = l2.indexer.wk(latent_enc)
    k_exp_enc = _cpu_rope(l2.indexer.k_norm(k_pre_enc), (pos_enc // 2) * 2, rope_cs)
    assert torch.equal(_read_k_slot(3, 0), k_exp_enc[1])
    assert torch.equal(_read_k_slot(3, 1), k_exp_enc[3])
    assert torch.equal(_read_k_slot(3, 2), sentinel_bf16)


def test_should_compile_with_zero_recompiles_and_zero_symint_mutations(
    cpu_dist_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    monkeypatch.setattr(DeepseekV41Attention, "kv_rope_op", staticmethod(_cpu_rope))
    monkeypatch.setattr(
        DeepseekV41Attention, "o_proj_op", staticmethod(_cpu_inverse_rope_wo_a)
    )
    monkeypatch.setattr(DeepseekV41Indexer, "kv_rope_op", staticmethod(_cpu_rope))
    monkeypatch.setattr(DeepseekCompressor, "kv_rope_op", staticmethod(_cpu_rope))
    monkeypatch.setattr(
        cache_mod.cache_op,
        "dsv41_write_sub_rows",
        lambda bank, _u, _s, after=None, page_tokens=16: (
            bank.clone() if after is None else bank + (after.amin() * 0).to(bank.dtype)
        ),
    )
    monkeypatch.setattr(
        cache_mod.swa_window_op,
        "dsv41_gather_swa_windows",
        lambda pool, page_ids, _phase, window_rows, **_kw: pool.new_zeros(
            (page_ids.shape[0], window_rows, ROW_WORDS), dtype=torch.int32
        ),
    )
    monkeypatch.setattr(
        dsv41_ops.attention_op,
        "dsv41_sparse_mla_swa_attention",
        lambda q, *_args: torch.zeros_like(q),
    )

    vcfg = _make_dsv41_vllm_config(
        block_size=4, sliding_window=4, index_topk=4, max_model_len=64
    )
    topk_buf = torch.full((16, 4), -1, dtype=torch.int32)
    with set_current_vllm_config(vcfg):
        layer = DeepseekV41Attention(vcfg, "model.layers.0.attn", topk_buf)
    _init_attention_weights(layer, seed=71, fp8_wo_a=True)
    layer.process_weights_after_loading(torch.bfloat16)
    raw_swa = torch.zeros((5, 4 * 2, ROW_WORDS), dtype=torch.int32)
    layer.swa_cache_layer.bind_kv_cache(
        torch.empty(0),
        raw_cache=raw_swa,
        layout=((5, 4, MAIN_HEAD_DIM), torch.bfloat16, 4 * MAIN_HEAD_DIM * 2, 0),
    )

    for non_causal in (False, True):
        layer.non_causal_multi_token_decode = non_causal
        layer.dspark_block_size = 2 if non_causal else 0
        counter = CompileCounter()
        captured_graphs: list[Any] = []

        def _capture_backend(
            gm: Any,
            example_inputs: Any,
            _graphs: list[Any] = captured_graphs,
            _ctr: CompileCounter = counter,
        ) -> Any:
            _graphs.append(gm)
            return _ctr(gm, example_inputs)

        def _step(positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            return layer(positions, hidden)

        compiled = torch.compile(_step, backend=_capture_backend, dynamic=True)
        for num_tokens in (4, 8):
            pos = torch.arange(num_tokens, dtype=torch.int32)
            hidden = torch.randn((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.1
            meta = _make_attn_meta(
                block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
                seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
                query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
                positions=pos,
            )
            with override_forward_context(
                _forward_ctx({layer.swa_cache_layer.prefix: meta})
            ):
                out = compiled(pos, hidden)
            assert out.shape == (num_tokens, HIDDEN_SIZE)
        assert counter.frame_count == 1
        if not non_causal:
            assert any(node.target == "amin" for node in captured_graphs[0].graph.nodes)
        for obj in (layer, layer.swa_cache_layer):
            for val in obj.__dict__.values():
                assert not isinstance(val, torch.SymInt)

    kv_graphs: list[Any] = []
    kv_counter = CompileCounter()

    def _capture_kv_backend(gm: Any, example_inputs: Any) -> Any:
        kv_graphs.append(gm)
        return kv_counter(gm, example_inputs)

    compiled_kv = torch.compile(
        lambda p, h: layer.forward_kv(p, h), backend=_capture_kv_backend, dynamic=True
    )
    pos_kv = torch.arange(4, dtype=torch.int32)
    h_kv = torch.randn((4, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.1
    meta_kv = _make_attn_meta(
        block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        positions=pos_kv,
    )
    with override_forward_context(
        _forward_ctx({layer.swa_cache_layer.prefix: meta_kv})
    ):
        compiled_kv(pos_kv, h_kv)
    assert kv_counter.frame_count == 1
    assert not any(node.target == "amin" for node in kv_graphs[0].graph.nodes)


@pytest.mark.tpu_test
def test_tpu_should_match_cpu_reference_on_physical_tpu7x(
    request: pytest.FixtureRequest,
) -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only test

    request.getfixturevalue("dist_ctx")
    dev = torch.device("tpu:0")
    vcfg = _make_dsv41_vllm_config(
        block_size=16,
        sliding_window=8,
        index_n_heads=4,
        index_topk=4,
        max_model_len=256,
        max_position_embeddings=256,
        max_num_seqs=4,
        device="tpu",
    )
    topk_tpu = torch.full((16, 4), -1, dtype=torch.int32, device=dev)
    cand_tpu = torch.full((16, 4), -1, dtype=torch.int32, device=dev)
    with set_current_vllm_config(vcfg):
        l0 = DeepseekV41Attention(vcfg, "model.layers.0.attn", topk_tpu)
        l2 = DeepseekV41Attention(vcfg, "model.layers.2.attn", topk_tpu, None, cand_tpu)
        l3 = DeepseekV41Attention(vcfg, "model.layers.3.attn", topk_tpu, None, cand_tpu)
        l14 = DeepseekV41Attention(
            vcfg, "model.layers.14.attn", topk_tpu, None, cand_tpu
        )
        l19 = DeepseekV41Attention(
            vcfg, "model.layers.19.attn", topk_tpu, None, cand_tpu
        )
        l20 = DeepseekV41Attention(
            vcfg, "model.layers.20.attn", topk_tpu, None, cand_tpu
        )

    all_layers = (l0, l2, l3, l14, l19, l20)
    for seed, lyr in enumerate(all_layers, start=81):
        _init_attention_weights(lyr, seed=seed, fp8_wo_a=True)
        lyr.process_weights_after_loading(torch.bfloat16)
        lyr.to(dev)

    runner = _RunnerTestHarness(vcfg, dev, max_num_reqs=4)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 20)
    mgr.initialize_kv_cache(kv_cfg)

    assert l20.paged_cache_rows is not None
    assert l20.paged_cache_rows.page_stride == l20.paged_cache_rows.shape[1]
    assert l0.swa_cache_layer.paged_cache_rows is not None
    assert l0.swa_cache_layer.paged_cache_rows.supports_paged_windows is True

    seq_lens_p = torch.tensor([8, 8], dtype=torch.int32, device=dev)
    qsl_p = torch.tensor([0, 8, 16], dtype=torch.int32, device=dev)
    pos_p = torch.tensor(list(range(8)) + list(range(8)), dtype=torch.int32, device=dev)
    meta_map_p: dict[str, AttentionMetadata] = {}
    next_bid = 1
    for group in kv_cfg.kv_cache_groups:
        cols = 1 if group.kv_cache_spec.block_size == 1 else 2
        g_bt = torch.arange(
            next_bid, next_bid + 2 * cols, dtype=torch.int32, device=dev
        ).reshape(2, cols)
        next_bid += 2 * cols
        g_meta = _make_attn_meta(g_bt, seq_lens_p, qsl_p, pos_p)
        for layer_name in group.layer_names:
            meta_map_p[layer_name] = g_meta
    for lyr in all_layers:
        if lyr.prefix not in meta_map_p and lyr.compressed_cache_prefix in meta_map_p:
            meta_map_p[lyr.prefix] = meta_map_p[lyr.compressed_cache_prefix]

    gen = torch.Generator().manual_seed(123)
    h_p = (torch.randn((16, HIDDEN_SIZE), generator=gen) * 0.25).bfloat16().to(dev)

    def _run_layers(
        positions: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y0 = l0(positions, hidden)
        y2 = l2(positions, hidden)
        _ = l3(positions, hidden)
        _ = l14(positions, hidden)
        _ = l19(positions, hidden)
        y20 = l20(positions, hidden)
        return y0, y2, y20

    torch.compiler.reset()
    compiled = torch.compile(_run_layers, backend="tpu", fullgraph=True, dynamic=False)
    with (
        set_vllm_model_wrapper_context(mesh=cast(Any, None), vllm_config=vcfg),
        override_forward_context(_forward_ctx(meta_map_p)),
    ):
        for lyr in all_layers:
            _ = lyr.kv_rope_op
            _ = lyr.o_proj_op
            if lyr.compressor is not None:
                _ = lyr.compressor.kv_rope_op
            if lyr.indexer is not None:
                _ = lyr.indexer.kv_rope_op
        y0_tpu, y2_tpu, y20_tpu = compiled(pos_p, h_p)

    assert torch.isfinite(y0_tpu.cpu()).all()
    assert torch.isfinite(y2_tpu.cpu()).all()
    assert torch.isfinite(y20_tpu.cpu()).all()
    assert y0_tpu.shape == (16, HIDDEN_SIZE)

    h_p_cpu = h_p.cpu()
    pos_p_cpu = pos_p.cpu()
    l0_w = {k: v.cpu() for k, v in l0.state_dict().items()}
    cos_sin_cpu = l0.rotary_emb.cos_sin_cache.cpu()
    qr_kv_ref = h_p_cpu.float() @ l0_w["fused_wqa_wkv.weight"].float().T
    qr_ref = rms_norm(qr_kv_ref[:, :Q_LORA_RANK], l0_w["q_norm.weight"].float(), l0.eps)
    kv_ref = rms_norm(
        qr_kv_ref[:, Q_LORA_RANK:], l0_w["kv_norm.weight"].float(), l0.eps
    )
    q_rot_ref = _cpu_rope(
        (qr_ref @ l0_w["wq_b.weight"].float().T)
        .bfloat16()
        .view(-1, NUM_HEADS, MAIN_HEAD_DIM),
        pos_p_cpu,
        cos_sin_cpu,
    )
    kv_rot_ref = _cpu_rope(kv_ref.bfloat16(), pos_p_cpu, cos_sin_cpu)
    mqa_ref = _reference_attention_fp32(
        q_rot_ref,
        [kv_rot_ref[:8], kv_rot_ref[8:16]],
        None,
        None,
        l0_w["attn_sink"],
        [8, 8],
        [0, 8, 16],
        sm_scale=l0.scale,
        window_size=l0.window_size,
    )
    z_ref = _cpu_inverse_rope_wo_a(
        mqa_ref.bfloat16(),
        pos_p_cpu,
        cos_sin_cpu,
        l0_w["wo_a.weight"],
        l0_w["wo_a.weight_scale"],
    )
    y0_ref = z_ref.float() @ l0_w["wo_b.weight"].float().T
    y0_got = y0_tpu.cpu().float()
    y0_cos = torch.nn.functional.cosine_similarity(
        y0_got.reshape(-1), y0_ref.reshape(-1), dim=0
    ).item()
    y0_max_diff = (y0_got - y0_ref).abs().max().item()
    assert y0_cos > 0.998, f"cos={y0_cos:.6f} max_diff={y0_max_diff:.4f}"
    assert torch.allclose(y0_got, y0_ref, atol=1.2e-1, rtol=1e-1), (
        f"max_diff={y0_max_diff:.4f} cos={y0_cos:.6f}"
    )

    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].cpu().abs().max().item()) == 0
    runner.zero_blocks_public([0, 1, 2])
    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].cpu().abs().max().item()) == 0
        assert int(raw_root[1].cpu().abs().max().item()) == 0
        assert int(raw_root[2].cpu().abs().max().item()) == 0
