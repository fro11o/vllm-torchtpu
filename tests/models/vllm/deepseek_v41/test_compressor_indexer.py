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
"""Unit and TPU7x tests for DeepSeek-V4.1 metadata, compressor, and indexer."""

from __future__ import annotations

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
from torch._dynamo.testing import CompileCounterWithBackend
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
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v41 import attention as upstream_attn
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.v1.attention.backend import CommonAttentionMetadata, MultipleOf
from vllm.v1.core import kv_cache_planning as kcu
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    ROW_BYTES,
    ROW_WORDS,
    DsV41CacheMixin,
    PagedCacheRows,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import cache as cache_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import indexer as idx_mod
from vllm_torchtpu.models.vllm.deepseek_v41 import metadata as meta_mod
from vllm_torchtpu.models.vllm.deepseek_v41.compressor import (
    AttentionBackend,
    CompressorStateCache,
    DeepseekCompressor,
    save_compress_norm,
)
from vllm_torchtpu.models.vllm.deepseek_v41.indexer import (
    INDEX_HEAD_DIM,
    DeepseekV41Indexer,
    IndexerCache,
    TpuSparseAttnIndexer,
    publish_candidate_blocks,
)
from vllm_torchtpu.models.vllm.deepseek_v41.metadata import (
    active_cache,
    block_table,
    cache_slots,
    get_kv_rope_op,
    layer_meta,
    not_replayed,
    query_token_requests,
    replay_starts,
    rotate_rope,
    token_requests,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

GIB_BYTES: int = 1 << 30
MAIN_HEAD_DIM: int = 512
ROPE_HEAD_DIM: int = 64
HIDDEN_SIZE: int = 64
Q_LORA_RANK: int = 32
SLIDING_WINDOW: int = 8
KEYS_PER_WORD_ROW: int = 2
BF16_SHIFT_BITS: int = 16
BF16_ODD_MASK: int = -65536


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


def _init_module_bf16(module: nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    module.to(torch.bfloat16)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" in name or "weight" not in name:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)


class _StubModelConfig:
    def __init__(self, hf_config: DeepseekV41Config, max_model_len: int) -> None:
        self.hf_config = hf_config
        self.hf_text_config = hf_config
        self.dtype = torch.bfloat16
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
    max_model_len: int = 256,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 64,
    index_n_heads: int = 4,
    index_topk: int = 8,
    tp_size: int = 1,
    device: str = "cpu",
) -> VllmConfig:
    hf_cfg = DeepseekV41Config(
        vocab_size=256,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=40,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        kv_lora_rank=MAIN_HEAD_DIM - ROPE_HEAD_DIM,
        q_lora_rank=Q_LORA_RANK,
        qk_nope_head_dim=64,
        qk_rope_head_dim=ROPE_HEAD_DIM,
        v_head_dim=64,
        head_dim=MAIN_HEAD_DIM,
        index_n_heads=index_n_heads,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=index_topk,
        sliding_window=SLIDING_WINDOW,
        max_position_embeddings=max_model_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        text_config={"rope_scaling": {"rope_type": "default"}},
    )
    cache_cfg = CacheConfig(block_size=block_size, cache_dtype="auto")
    cache_cfg.kv_cache_layout = "BLHNC"
    vcfg = VllmConfig(
        cache_config=cache_cfg,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device=device),
        attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=5),
    )
    vcfg.parallel_config.tensor_parallel_size = tp_size
    vcfg.model_config = cast(Any, _StubModelConfig(hf_cfg, max_model_len))
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


class _DummyMlaCacheLayer(DsV41CacheMixin, AttentionLayerBase):
    def __init__(self, spec: KVCacheSpec | None) -> None:
        super().__init__()
        self.spec = spec
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[Any]:
        return AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        del vllm_config
        return self.spec


class _PlainDenseCacheLayer(nn.Module):
    def __init__(self, kv_cache: torch.Tensor) -> None:
        super().__init__()
        self.kv_cache = kv_cache


class _DummyRotaryEmb(nn.Module):
    def __init__(self, max_pos: int = 128, rope_dim: int = ROPE_HEAD_DIM) -> None:
        super().__init__()
        self.cos_sin_cache = torch.arange(
            max_pos * rope_dim, dtype=torch.float32
        ).reshape(max_pos, rope_dim)


def _identity_rope_op(
    inputs: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    del cos_sin_cache
    out = inputs.clone()
    out[..., -1] = out[..., -1] + positions.view(
        positions.shape[0], *([1] * (inputs.ndim - 2))
    ).to(inputs.dtype)
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
        self.input_batch = SimpleNamespace(block_table=[])
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
        self.input_batch = SimpleNamespace(block_table=tables)

    def set_builder_context(self, ctx: AttentionMetadataBuilderContext) -> None:
        self._attn_metadata_builder_ctx = ctx

    def zero_blocks_public(self, block_ids: list[int]) -> None:
        self._zero_block_ids(block_ids)


@pytest.fixture
def cpu_cache_ops(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    write_after_flags: list[bool] = []

    def _gather(pool: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        pool_2d = pool.reshape(-1, ROW_WORDS)
        valid = (idx >= 0) & (idx < pool_2d.shape[0])
        safe = idx.masked_fill(~valid, 0).long()
        return pool_2d.index_select(0, safe).masked_fill(~valid.unsqueeze(-1), 0)

    def _write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        *,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        del page_tokens
        write_after_flags.append(after is not None)
        out = bank.clone()
        sub = out.view(-1, updates.shape[1])
        loc = slots.long()
        valid = (loc >= 0) & (loc < sub.shape[0])
        sub[loc[valid]] = updates[valid]
        return out

    def _pack_fp8(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = latents[:, 0].float().nan_to_num(0.0).to(torch.int32)
        nope = base.unsqueeze(1).expand(-1, ROW_WORDS).contiguous()
        rope = (base + 1000).unsqueeze(1).expand(-1, 32).contiguous()
        return nope, rope

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_gather_cache_rows", _gather)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _write)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_pack_fp8_kv_rows", _pack_fp8)
    return write_after_flags


def _unpack_split_bf16_keys(words: torch.Tensor) -> torch.Tensor:
    keys_words = words.reshape(-1, INDEX_HEAD_DIM // 2)
    even = (keys_words << BF16_SHIFT_BITS).view(torch.float32).to(torch.bfloat16)
    odd = (keys_words & BF16_ODD_MASK).view(torch.float32).to(torch.bfloat16)
    return torch.cat((even, odd), dim=-1).float()


def _ref_paged_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    topk: int,
    candidate_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    num_tokens = q.shape[0]
    page_rows = k_pool.shape[1]
    keys_per_page = KEYS_PER_WORD_ROW * page_rows
    pool_keys = _unpack_split_bf16_keys(k_pool).view(
        k_pool.shape[0], keys_per_page, INDEX_HEAD_DIM
    )
    out = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=q.device)
    num_seqs = seq_lens.shape[0]
    for s in range(num_seqs):
        start = int(cu_q_lens[s].item())
        end = int(cu_q_lens[s + 1].item())
        q_len = end - start
        if q_len <= 0:
            continue
        seq_len = int(seq_lens[s].item())
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
            q_row = q[row_i].float()
            w_row = weights[row_i].float().squeeze(-1)
            for kp in valid_pos:
                page_col = kp // keys_per_page
                in_page = kp % keys_per_page
                phys_page = int(block_tables[s, page_col].item())
                k_vec = pool_keys[phys_page, in_page]
                dot = torch.relu(q_row @ k_vec)
                score = float((dot * w_row).sum().item())
                scored.append((score, kp))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            chosen = [kp for _, kp in scored[:topk]]
            out[row_i, : len(chosen)] = torch.tensor(
                chosen, dtype=torch.int32, device=q.device
            )
    return out


def test_should_route_metadata_slots_rope_and_replay_windows_when_batched() -> None:
    rotary = _DummyRotaryEmb(max_pos=10, rope_dim=8)
    seen_pos: list[list[int]] = []

    def _capture_rope(
        inputs: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
    ) -> torch.Tensor:
        assert cos_sin is rotary.cos_sin_cache
        seen_pos.append(positions.tolist())
        return inputs

    raw_in = torch.zeros((4, 8), dtype=torch.bfloat16)
    rotate_rope(
        raw_in, torch.tensor([-5, 0, 9, 99], dtype=torch.int32), rotary, _capture_rope
    )
    assert seen_pos == [[0, 0, 9, 9]]

    # active_cache returns PagedCacheRows when bound, else dense kv_cache tensor.
    dense_layer = _PlainDenseCacheLayer(torch.ones((2, 4), dtype=torch.bfloat16))
    assert active_cache(dense_layer) is dense_layer.kv_cache

    mla_layer = _DummyMlaCacheLayer(None)
    assert active_cache(mla_layer) is mla_layer.kv_cache
    raw_pool = torch.zeros((4, 4, ROW_WORDS), dtype=torch.int32)
    mla_layer.bind_kv_cache(
        raw_pool,
        raw_cache=raw_pool,
        layout=((4, 4, 256), torch.bfloat16, 4 * ROW_BYTES, 0),
    )
    pcr_null = active_cache(mla_layer)
    assert isinstance(pcr_null, PagedCacheRows) and pcr_null.null_block is True

    # layer_meta handles both dict and direct AttentionMetadata payloads.
    meta_a = _make_attn_meta(
        block_tables=torch.tensor([[1, 2], [0, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([5, 3], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    with override_forward_context(_forward_ctx({"layer.a": meta_a})):
        assert layer_meta("layer.a") is meta_a
        assert layer_meta("missing") is None
        assert replay_starts(2, torch.device("cpu")) is None
        assert not_replayed(meta_a, torch.arange(4, dtype=torch.int32), 4).all()
    with override_forward_context(_forward_ctx(meta_a)):
        assert layer_meta("any") is meta_a

    # block_table & query_token_requests / token_requests.
    assert block_table(meta_a).tolist() == [[1, 2], [0, 3]]
    positions = torch.tensor([0, 3, 4, 0, -1, 0], dtype=torch.int32)
    assert query_token_requests(meta_a.query_start_loc, positions).tolist() == [
        0,
        0,
        0,
        1,
        2,
        2,
    ]
    assert token_requests(meta_a, positions).tolist() == [0, 0, 0, 1, 2, 2]

    # Paged slots with null_block=True mask block 0, negative pos, and padded tokens.
    slots_c1 = cache_slots(pcr_null, meta_a, positions, compress_ratio=1, ring=False)
    assert slots_c1.tolist() == [1 * 4 + 0, 1 * 4 + 3, 2 * 4 + 0, -1, -1, -1]

    # OOB raw_cols >= num_cols must return -1 rather than clamping to last block col.
    oob_slots = cache_slots(
        pcr_null,
        meta_a,
        torch.tensor([8, 0, 0, 0], dtype=torch.int32),
        compress_ratio=1,
        ring=False,
    )
    assert oob_slots[0].item() == -1

    # Paged slots with null_block=False allow block 0; compress_ratio=2 scales pos.
    pcr_nonull = PagedCacheRows(
        raw_pool, ((2, 4, 256), torch.bfloat16, 2048, 0), null_block=False
    )
    slots_c2 = cache_slots(pcr_nonull, meta_a, positions, compress_ratio=2, ring=False)
    assert slots_c2.tolist() == [1 * 4 + 0, 1 * 4 + 1, 1 * 4 + 2, 0 * 4 + 0, -1, -1]

    # Ring slots wrap modulo rows_per_page on col 0 and mask null_block=0 and pad rows.
    ring_pos = torch.tensor([0, 7, 9, 2, -1, 0], dtype=torch.int32)
    slots_ring = cache_slots(pcr_null, meta_a, ring_pos, ring=True)
    assert slots_ring.tolist() == [1 * 4 + 0, 1 * 4 + 3, 1 * 4 + 1, -1, -1, -1]

    # Empty seq_lens or zero-width block_tables return all -1.
    empty_seq_meta = _make_attn_meta(
        torch.zeros((2, 2), dtype=torch.int32),
        torch.empty((0,), dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    assert cache_slots(pcr_null, empty_seq_meta, positions).tolist() == [-1] * 6
    zero_col_meta = _make_attn_meta(
        torch.empty((2, 0), dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert cache_slots(pcr_null, zero_col_meta, positions).tolist() == [-1] * 6

    # Replay starts & not_replayed coverage across window and decoder starts.
    win_t = torch.tensor([0, 10], dtype=torch.int32)
    dec_t = torch.tensor([4, 6], dtype=torch.int32)
    with override_forward_context(
        _forward_ctx(
            meta_a,
            {
                "deepseek_v41_window_starts": torch.empty(0, dtype=torch.int32),
                "deepseek_v41_replay_starts": dec_t,
            },
        )
    ):
        rs_enc = replay_starts(2, torch.device("cpu"), ced_decoder=False)
        rs_dec = replay_starts(2, torch.device("cpu"), ced_decoder=True)
        assert rs_enc is not None and rs_enc.tolist() == [4, 6]
        assert rs_dec is not None and rs_dec.tolist() == [4, 6]
    with override_forward_context(
        _forward_ctx(meta_a, {"deepseek_v41_window_starts": win_t})
    ):
        rs_win_enc = replay_starts(2, torch.device("cpu"), ced_decoder=False)
        rs_win_dec = replay_starts(2, torch.device("cpu"), ced_decoder=True)
        assert rs_win_enc is not None and rs_win_enc.tolist() == [0, 10]
        assert rs_win_dec is not None and rs_win_dec.tolist() == [0, 10]
    with override_forward_context(
        _forward_ctx(
            meta_a,
            {"deepseek_v41_window_starts": win_t, "deepseek_v41_replay_starts": dec_t},
        )
    ):
        rs_both_enc = replay_starts(2, torch.device("cpu"), ced_decoder=False)
        rs_both_dec = replay_starts(2, torch.device("cpu"), ced_decoder=True)
        assert rs_both_enc is not None and rs_both_enc.tolist() == [0, 10]
        assert rs_both_dec is not None and rs_both_dec.tolist() == [4, 10]
        rep_meta = _make_attn_meta(
            torch.ones((2, 2), dtype=torch.int32),
            torch.tensor([20, 20], dtype=torch.int32),
            torch.tensor([0, 2, 6], dtype=torch.int32),
        )
        rep_pos = torch.tensor([0, 2, 9, 10, 13, 14], dtype=torch.int32)
        mask = not_replayed(rep_meta, rep_pos, window_size=4)
        # Req 0 has start=0 (all writable); req 1 has start=10 -> [10, 14) skipped.
        assert mask.tolist() == [True, True, True, False, False, True]


def test_should_cache_and_trace_get_kv_rope_op_across_mesh_modes(
    cpu_dist_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    recorded: list[dict[str, object]] = []
    real_jax_op = tpu_pallas.jax_op

    def _spy_jax_op(name: str, fn: Any, **kwargs: object) -> Any:
        recorded.append({"name": name, **kwargs})
        return real_jax_op(name, fn, **kwargs)

    monkeypatch.setattr(tpu_pallas, "jax_op", _spy_jax_op)

    def _ref_rope_jax(
        x: jax.Array, positions: jax.Array, cos_sin_cache: jax.Array
    ) -> jax.Array:
        scale = cos_sin_cache[positions, :1]
        if x.ndim == 3:
            scale = scale[:, None, :]
        return (x.astype(jnp.float32) + scale).astype(x.dtype)

    monkeypatch.setattr(meta_mod, "rope", _ref_rope_jax)

    single_mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    with set_vllm_model_wrapper_context(mesh=single_mesh):
        op_single = get_kv_rope_op()
        assert get_kv_rope_op() is op_single
        if recorded:
            assert recorded[-1]["mesh"] is None
            assert recorded[-1]["input_partition_specs"] is None

        # Verify JAX bridge forwarding without compiling Pallas on CPU.
        x_np = np.arange(2 * 128, dtype=np.float32).reshape(2, 128)
        pos_np = np.array([1, 3], dtype=np.int32)
        cs_np = np.arange(8 * 64, dtype=np.float32).reshape(8, 64)
        out = op_single(
            _JaxBackedTensor(jnp.asarray(x_np, dtype=jnp.bfloat16)),
            _JaxBackedTensor(jnp.asarray(pos_np, dtype=jnp.int32)),
            _JaxBackedTensor(jnp.asarray(cs_np, dtype=jnp.float32)),
        )
        assert isinstance(out, _JaxBackedTensor)
        expected = _ref_rope_jax(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(pos_np, dtype=jnp.int32),
            jnp.asarray(cs_np, dtype=jnp.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(out.to_numpy(), dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
        )

        # Verify FakeTensor + fullgraph torch.compile tracing for 2D & 3D shapes.
        counter = CompileCounterWithBackend("aot_eager")
        compiled = torch.compile(
            lambda x, pos, cs: op_single(x, pos, cs),
            backend=counter,
            fullgraph=True,
        )
        with FakeTensorMode():
            out_2d = compiled(
                torch.empty((4, MAIN_HEAD_DIM), dtype=torch.bfloat16),
                torch.empty((4,), dtype=torch.int32),
                torch.empty((16, ROPE_HEAD_DIM), dtype=torch.float32),
            )
            assert out_2d.shape == (4, MAIN_HEAD_DIM)
            assert out_2d.dtype == torch.bfloat16
            out_3d = op_single(
                torch.empty((4, 4, INDEX_HEAD_DIM), dtype=torch.bfloat16),
                torch.empty((4,), dtype=torch.int32),
                torch.empty((16, ROPE_HEAD_DIM), dtype=torch.float32),
            )
            assert out_3d.shape == (4, 4, INDEX_HEAD_DIM)
        assert counter.frame_count == 1

        # Verify DeepseekCompressor & DeepseekV41Indexer expose kv_rope_op properties.
        assert hasattr(DeepseekCompressor, "kv_rope_op")
        assert hasattr(DeepseekV41Indexer, "kv_rope_op")
        vcfg = _make_dsv41_vllm_config(block_size=16, index_n_heads=4, index_topk=4)
        with set_current_vllm_config(vcfg):
            comp = DeepseekCompressor(
                vllm_config=vcfg,
                compress_ratio=1,
                hidden_size=HIDDEN_SIZE,
                head_dim=MAIN_HEAD_DIM,
                prefix="model.layers.20.attn.compressor",
                k_cache_prefix="model.layers.20.attn",
            )
            idx_cache = IndexerCache(
                INDEX_HEAD_DIM,
                torch.bfloat16,
                "model.layers.20.attn.indexer.k_cache",
                vcfg.cache_config,
                compress_ratio=1,
            )
            idx = DeepseekV41Indexer(
                vllm_config=vcfg,
                config=vcfg.model_config.hf_config,
                hidden_size=HIDDEN_SIZE,
                q_lora_rank=Q_LORA_RANK,
                quant_config=None,
                cache_config=vcfg.cache_config,
                topk_indices_buffer=None,
                compress_ratio=1,
                prefix="model.layers.20.attn.indexer",
                owns_k=False,
                k_cache=idx_cache,
                main_head_dim=MAIN_HEAD_DIM,
            )
        assert comp.kv_rope_op is op_single
        assert idx.kv_rope_op is op_single

    multi_mesh = AbstractMesh((2,), ("data",))
    # Avoid world_size=1 mismatch from cpu_dist_ctx on a 2-device AbstractMesh.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with set_vllm_model_wrapper_context(mesh=cast(Any, multi_mesh)):
        op_multi = get_kv_rope_op()
        assert get_kv_rope_op() is op_multi
        assert op_multi is not op_single
        if recorded:
            assert recorded[-1]["mesh"] is multi_mesh
            assert recorded[-1]["input_partition_specs"] == (P(), P(), P())
        with FakeTensorMode():
            out_m = op_multi(
                torch.empty((2, MAIN_HEAD_DIM), dtype=torch.bfloat16),
                torch.empty((2,), dtype=torch.int32),
                torch.empty((16, ROPE_HEAD_DIM), dtype=torch.float32),
            )
            assert out_m.shape == (2, MAIN_HEAD_DIM)


def _oracle_pair_compress(
    kv_score_even: torch.Tensor,
    kv_score_odd: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    head_dim = norm_weight.shape[0]
    kv0, s0 = kv_score_even[:, :head_dim].float(), kv_score_even[:, head_dim:].float()
    kv1, s1 = kv_score_odd[:, :head_dim].float(), kv_score_odd[:, head_dim:].float()
    gates = torch.softmax(torch.stack((s0, s1), dim=1), dim=1)
    pooled = gates[:, 0] * kv0 + gates[:, 1] * kv1
    return rms_norm(pooled, norm_weight.float(), eps).to(torch.bfloat16)


def test_should_compress_cold_and_warm_cross_step_pairs_when_running_compressor(
    cpu_dist_ctx: None,
    cpu_cache_ops: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx, cpu_cache_ops
    vcfg_none_bs = _make_dsv41_vllm_config(block_size=None)
    assert AttentionBackend.get_page_size(vcfg_none_bs) == 16
    assert AttentionBackend.get_min_page_size(vcfg_none_bs) == 1
    assert AttentionBackend.get_supported_kernel_block_sizes() == [MultipleOf(1)]
    assert AttentionBackend.supported_kv_cache_layouts() == (KVCacheLayout.BLHNC,)

    vcfg = _make_dsv41_vllm_config(block_size=32)
    assert AttentionBackend.get_page_size(vcfg) == 32
    comp_rope_calls: list[list[int]] = []

    def _spy_comp_rope(
        inputs: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        comp_rope_calls.append(positions.tolist())
        return _identity_rope_op(inputs, positions, cos_sin_cache)

    monkeypatch.setattr(DeepseekCompressor, "kv_rope_op", staticmethod(_spy_comp_rope))

    with set_current_vllm_config(vcfg):
        comp2 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=2,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.2.attn.compressor",
            k_cache_prefix="model.layers.2.attn",
        )
        comp1 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=1,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.20.attn.compressor",
            k_cache_prefix="model.layers.20.attn",
        )
        real_rotary = upstream_attn.build_deepseek_v4_rope(
            vcfg.model_config.hf_config,
            head_dim=MAIN_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            max_position_embeddings=256,
            compress_ratio=1,
        )
    assert real_rotary.cos_sin_cache.shape == (256, ROPE_HEAD_DIM)
    _init_module_bf16(comp2, 11)
    _init_module_bf16(comp1, 12)

    assert isinstance(comp2.state_cache, CompressorStateCache)
    assert isinstance(comp2.state_cache, DsV41CacheMixin)
    assert comp2.state_cache.get_attn_backend() is AttentionBackend
    assert (
        vcfg.compilation_config.static_forward_context[
            "model.layers.2.attn.compressor.state_cache"
        ]
        is comp2.state_cache
    )
    assert comp1.state_cache is None

    # Before binding state_cache (numel == 0), forward falls back to per-token RMSNorm.
    gen = torch.Generator().manual_seed(42)
    hidden = torch.randn((8, HIDDEN_SIZE), generator=gen, dtype=torch.bfloat16)
    kv_score_all = comp2.project(hidden)
    assert kv_score_all.shape == (8, 2 * MAIN_HEAD_DIM)
    assert kv_score_all.dtype == torch.float32

    # TP > 1 replicates compressor projections identically to TP = 1.
    vcfg_tp4 = _make_dsv41_vllm_config(block_size=32, tp_size=4)
    with set_current_vllm_config(vcfg_tp4):
        comp2_tp4 = DeepseekCompressor(
            vllm_config=vcfg_tp4,
            compress_ratio=2,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.2.attn.compressor",
            k_cache_prefix="model.layers.2.attn",
        ).to(torch.bfloat16)
    comp2_tp4.load_state_dict(comp2.state_dict())
    assert torch.equal(comp2_tp4.project(hidden), kv_score_all)

    with override_forward_context(_forward_ctx(None)):
        unbound_out = comp2(kv_score_all[:2], torch.tensor([0, 1], dtype=torch.int32))
        assert torch.allclose(
            unbound_out.float(),
            rms_norm(kv_score_all[:2, :MAIN_HEAD_DIM], comp2.norm.weight.float(), 1e-6),
            atol=1e-2,
        )

    # Bind TPU PagedCacheRows to comp2.state_cache (ring_cap=8, state_dim=1024 fp32).
    ring_cap = comp2.state_cache.block_size
    state_raw = torch.zeros((4, ring_cap * 8, ROW_WORDS), dtype=torch.int32)
    comp2.state_cache.bind_kv_cache(
        torch.empty(0),
        raw_cache=state_raw,
        layout=(
            (4, ring_cap, 2 * MAIN_HEAD_DIM),
            torch.float32,
            ring_cap * 8 * ROW_BYTES,
            0,
        ),
    )

    # Step 1 (Cold ring): pos [0..4] on req 0 (block 1) -> pairs (0,1),(2,3), saves 4.
    pos_step1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    meta_step1 = _make_attn_meta(
        block_tables=torch.tensor([[1]], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
    )
    ctx_map1 = {"model.layers.2.attn.compressor.state_cache": meta_step1}
    with override_forward_context(_forward_ctx(ctx_map1)):
        lat_step1 = comp2(kv_score_all[:5], pos_step1)

    oracle_01 = _oracle_pair_compress(
        kv_score_all[0:1], kv_score_all[1:2], comp2.norm.weight, comp2.rms_norm_eps
    )
    oracle_23 = _oracle_pair_compress(
        kv_score_all[2:3], kv_score_all[3:4], comp2.norm.weight, comp2.rms_norm_eps
    )
    assert torch.allclose(lat_step1[1:2], oracle_01, atol=1e-2, rtol=1e-2)
    assert torch.allclose(lat_step1[3:4], oracle_23, atol=1e-2, rtol=1e-2)

    # Step 2 (Warm cross-step odd start): pos [5, 6, 7] pairs token 5 with cached 4.
    pos_step2 = torch.tensor([5, 6, 7], dtype=torch.int32)
    meta_step2 = _make_attn_meta(
        block_tables=torch.tensor([[1]], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
    )
    ctx_map2 = {"model.layers.2.attn.compressor.state_cache": meta_step2}
    with override_forward_context(_forward_ctx(ctx_map2)):
        lat_step2 = comp2(kv_score_all[5:8], pos_step2)

    oracle_45 = _oracle_pair_compress(
        kv_score_all[4:5], kv_score_all[5:6], comp2.norm.weight, comp2.rms_norm_eps
    )
    oracle_67 = _oracle_pair_compress(
        kv_score_all[6:7], kv_score_all[7:8], comp2.norm.weight, comp2.rms_norm_eps
    )
    assert torch.allclose(lat_step2[0:1], oracle_45, atol=1e-2, rtol=1e-2)
    assert torch.allclose(lat_step2[2:3], oracle_67, atol=1e-2, rtol=1e-2)

    # Test comp1 and insert_cache across None, missing layer, empty cache, and writes.
    rotary = _DummyRotaryEmb(max_pos=64)
    comp2.insert_cache(None, pos_step1, rotary)
    with override_forward_context(_forward_ctx({})):
        comp2.insert_cache(lat_step1, pos_step1, rotary)

    mla_layer2 = _DummyMlaCacheLayer(None)
    vcfg.compilation_config.static_forward_context["model.layers.2.attn"] = mla_layer2
    mla_meta2 = _make_attn_meta(
        block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
    )
    with override_forward_context(_forward_ctx({"model.layers.2.attn": mla_meta2})):
        comp2.insert_cache(lat_step1, pos_step1, rotary)

    # Bind FP8 PagedCacheRows to model.layers.2.attn and check T // 2 writes.
    mla_nope = torch.zeros((4, 16, ROW_WORDS), dtype=torch.int32)
    mla_rope = torch.zeros((4, 4, ROW_WORDS), dtype=torch.int32)
    mla_layer2.bind_kv_cache(
        (mla_nope, mla_rope),
        raw_cache=mla_nope,
        layout=((4, 16, MAIN_HEAD_DIM), torch.uint8, 16 * ROW_BYTES, 0),
        rope_pool=mla_rope,
    )
    pos_6 = torch.arange(6, dtype=torch.int32)
    lat_6 = (
        torch.arange(1, 7, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(6, MAIN_HEAD_DIM)
        .contiguous()
    )
    with override_forward_context(_forward_ctx({"model.layers.2.attn": mla_meta2})):
        comp2.insert_cache(lat_6, pos_6, rotary)
    # Positions 0..5 with compress_ratio=2 write ONLY odd boundary tokens 1, 3, 5
    # and rotate compressed group j at position (pos // 2) * 2 = [0, 0, 2, 2, 4, 4].
    written_rows = mla_nope[1, :, 0].tolist()
    assert written_rows[:4] == [2, 4, 6, 0]
    assert comp_rope_calls[-1] == [0, 0, 2, 2, 4, 4]

    # Ratio-1 compressor forward + insert_cache writes all T active non-replayed tokens.
    mla_layer20 = _DummyMlaCacheLayer(None)
    mla20_nope = torch.zeros((4, 16, ROW_WORDS), dtype=torch.int32)
    mla20_rope = torch.zeros((4, 4, ROW_WORDS), dtype=torch.int32)
    mla_layer20.bind_kv_cache(
        (mla20_nope, mla20_rope),
        raw_cache=mla20_nope,
        layout=((4, 16, MAIN_HEAD_DIM), torch.uint8, 16 * ROW_BYTES, 0),
        rope_pool=mla20_rope,
    )
    vcfg.compilation_config.static_forward_context["model.layers.20.attn"] = mla_layer20
    with override_forward_context(_forward_ctx({"model.layers.20.attn": mla_meta2})):
        kv1 = comp1.project(hidden[:6])
        lat1 = comp1(kv1, pos_6)
        assert lat1.shape == (6, MAIN_HEAD_DIM)
        comp1.insert_cache(lat_6, pos_6, rotary)
    assert mla20_nope[1, :7, 0].tolist() == [1, 2, 3, 4, 5, 6, 0]
    assert comp_rope_calls[-1] == [0, 1, 2, 3, 4, 5]

    # Adjacent positions across a request boundary (req 0 pos=4 -> req 1 pos=5) must
    # NOT pair in-chunk across requests; req 1 must read its own ring state at slot 4.
    boundary_state = torch.zeros((4, 8, 2 * MAIN_HEAD_DIM), dtype=torch.float32)
    boundary_state[2, 4, :MAIN_HEAD_DIM] = 1.25
    boundary_state[2, 4, MAIN_HEAD_DIM:] = 8.0
    kv_boundary = torch.zeros((2, 2 * MAIN_HEAD_DIM), dtype=torch.float32)
    kv_boundary[0, :MAIN_HEAD_DIM] = -9.0
    kv_boundary[0, MAIN_HEAD_DIM:] = 10.0
    kv_boundary[1, :MAIN_HEAD_DIM] = 0.5
    kv_boundary[1, MAIN_HEAD_DIM:] = -8.0
    boundary_out = save_compress_norm(
        kv_boundary,
        torch.tensor([4, 5], dtype=torch.int32),
        boundary_state,
        torch.tensor([1 * 8 + 4, 2 * 8 + 5], dtype=torch.int32),
        comp2.norm.weight,
        comp2.rms_norm_eps,
        2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    oracle_req1 = _oracle_pair_compress(
        boundary_state[2, 4:5], kv_boundary[1:2], comp2.norm.weight, comp2.rms_norm_eps
    )
    assert torch.allclose(boundary_out[1:2], oracle_req1, atol=1e-2, rtol=1e-2)

    # Dense Tensor fallback must ignore negative (-1) slot_mapping on even rows.
    dense_neg = torch.zeros((3, 8, 2 * MAIN_HEAD_DIM), dtype=torch.float32)
    save_compress_norm(
        kv_score_all[:2],
        torch.tensor([0, 2], dtype=torch.int32),
        dense_neg,
        torch.tensor([-1, -1], dtype=torch.int32),
        comp2.norm.weight,
        comp2.rms_norm_eps,
        2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    assert torch.equal(dense_neg, torch.zeros_like(dense_neg))

    # Direct save_compress_norm with query_start_loc=None and dense state_cache tensor.
    dense_state = torch.zeros((3, 8, 2 * MAIN_HEAD_DIM), dtype=torch.float32)
    direct_out = save_compress_norm(
        kv_score_all[:4],
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        dense_state,
        torch.tensor([8, 9, 10, 11], dtype=torch.int32),
        comp2.norm.weight,
        comp2.rms_norm_eps,
        2,
        query_start_loc=None,
    )
    assert torch.allclose(direct_out[1:2], oracle_01, atol=1e-2, rtol=1e-2)
    assert torch.allclose(direct_out[3:4], oracle_23, atol=1e-2, rtol=1e-2)


def test_should_propagate_cross_step_nan_without_contaminating_even_or_in_chunk_rows(
    cpu_cache_ops: list[bool],
) -> None:
    del cpu_cache_ops
    head_dim = 16
    ring_cap = 8
    dense_state = torch.zeros((4, ring_cap, 2 * head_dim), dtype=torch.float32)
    dense_state[1].fill_(float("nan"))

    kv_score = torch.ones((4, 2 * head_dim), dtype=torch.float32) * 1.5
    positions = torch.tensor([2, 3, 1, 1], dtype=torch.int32)
    slot_mapping = torch.tensor(
        [1 * ring_cap + 2, 1 * ring_cap + 3, 2 * ring_cap + 1, 1 * ring_cap + 1],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    norm_w = torch.ones(head_dim, dtype=torch.bfloat16)

    out = save_compress_norm(
        kv_score,
        positions,
        dense_state,
        slot_mapping,
        norm_w,
        1e-6,
        compress_ratio=2,
        query_start_loc=query_start_loc,
    )
    assert torch.isfinite(out[:3]).all()
    assert torch.isnan(out[3]).all()


def test_should_select_dsv41_allocator_and_skip_dsv4_when_real_caches_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vcfg = _make_dsv41_vllm_config(block_size=16)
    with pytest.raises(ValueError, match="IndexerCache requires head_dim=128"):
        IndexerCache(64, torch.bfloat16, "bad.dim", vcfg.cache_config)
    with pytest.raises(ValueError, match="IndexerCache requires head_dim=128"):
        IndexerCache(INDEX_HEAD_DIM, torch.float16, "bad.dtype", vcfg.cache_config)

    with set_current_vllm_config(vcfg):
        idx_l2 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.2.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l20 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.20.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=1,
        )
        state_l2 = CompressorStateCache(
            state_dim=2 * MAIN_HEAD_DIM,
            dtype=torch.float32,
            prefix="model.layers.2.attn.compressor.state_cache",
        )
        mla_l20 = _DummyMlaCacheLayer(
            MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=MAIN_HEAD_DIM,
                dtype=torch.uint8,
                tokens_per_state=1,
                cache_dtype_str="fp8",
                alignment=ROW_BYTES,
                model_version="deepseek_v4",
                state_content_bytes=640,
            )
        )
        vcfg.compilation_config.static_forward_context["model.layers.20.attn"] = mla_l20

    assert idx_l2.get_attn_backend() is AttentionBackend
    raw_idx_spec = idx_l2.get_kv_cache_spec(vcfg)
    assert isinstance(raw_idx_spec, MLAAttentionSpec)
    assert raw_idx_spec.alignment == ROW_BYTES
    assert raw_idx_spec.page_size_padded is None

    dsv4_spy = MagicMock(
        side_effect=AssertionError("DsV4KVCacheAllocator must not run")
    )
    monkeypatch.setattr(
        TpuPlatform, "_find_non_ssm_backend", lambda *_a: PallasAttentionBackend
    )

    runner = _RunnerTestHarness(vcfg, torch.device("cpu"))
    mgr = KVCacheManager(runner)
    monkeypatch.setattr(mgr.ds_v4, "_initialize_ds_v4_kv_cache", dsv4_spy)
    specs = mgr.get_kv_cache_spec()
    assert set(specs.keys()) == {
        "model.layers.2.attn.indexer.k_cache",
        "model.layers.20.attn.indexer.k_cache",
        "model.layers.2.attn.compressor.state_cache",
        "model.layers.20.attn",
    }
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 4)
    mgr.initialize_kv_cache(kv_cfg)

    dsv4_spy.assert_not_called()
    assert runner.ds_v41_zero_owner is not None
    assert idx_l2.paged_cache_rows is not None
    assert idx_l20.paged_cache_rows is not None
    assert state_l2.paged_cache_rows is not None


def test_should_preserve_3d_pool_identity_and_dyadic_subpages_and_candidates_in_indexer(
    cpu_dist_ctx: None,
    cpu_cache_ops: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx, cpu_cache_ops
    # 1. publish_candidate_blocks: use_all, tail pin, dedup, truncation, and pad_cols.
    cand = publish_candidate_blocks(
        topk_indices=torch.tensor(
            [
                [0, 1, 2, -1],
                [18, 18, 4, 0],
                [0, 1, 2, 3],
            ],
            dtype=torch.int32,
        ),
        positions=torch.tensor([3, 39, 10], dtype=torch.int32),
        query_end=torch.tensor(2, dtype=torch.int32),
        compress_ratio=2,
        candidate_block_size=4,
        candidate_topk_blocks=4,
    )
    assert cand[0].tolist() == [0, -1, -1, -1]
    assert cand[1].tolist() == [4, 0, 1, -1]
    assert cand[2].tolist() == [-1, -1, -1, -1]

    cand_wide = publish_candidate_blocks(
        torch.tensor([[0, 8]], dtype=torch.int32),
        positions=torch.tensor([63], dtype=torch.int32),
        query_end=torch.tensor(1, dtype=torch.int32),
        compress_ratio=2,
        candidate_block_size=4,
        candidate_topk_blocks=6,
    )
    assert cand_wide.tolist() == [[7, 0, 2, -1, -1, -1]]

    # Tight use_all boundary (num_blocks == K vs K + 1), dedup, and truncation policy:
    # row 0: num_blocks=3 == K=3 -> all blocks [0, 1, 2];
    # row 1: num_blocks=4 > K=3 (tail=3) -> [3, 0, 1];
    # row 2: num_blocks=10 > K=3 (tail=9, unsorted blocks 7,6,1,0) -> [9, 0, 1];
    # row 3: repeated non-tail block 0 (0..3) + block 1 (4..5) -> deduped [9, 0, 1].
    cand_trunc = publish_candidate_blocks(
        torch.tensor(
            [
                [8, 8, -1, -1, -1, -1],
                [0, 4, 8, 12, -1, -1],
                [28, 24, 4, 0, 36, -1],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.int32,
        ),
        positions=torch.tensor([11, 12, 39, 39], dtype=torch.int32),
        query_end=torch.tensor(4, dtype=torch.int32),
        compress_ratio=1,
        candidate_block_size=4,
        candidate_topk_blocks=3,
    )
    assert cand_trunc.tolist() == [
        [0, 1, 2],
        [3, 0, 1],
        [9, 0, 1],
        [9, 0, 1],
    ]

    # 2. IndexerCache fp8_ds_mla alignment & TpuSparseAttnIndexer fail-fast.
    vcfg_fp8 = _make_dsv41_vllm_config(block_size=16, index_n_heads=4, index_topk=4)
    vcfg_fp8.cache_config.cache_dtype = "fp8_ds_mla"
    with set_current_vllm_config(vcfg_fp8):
        idx_fp8 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.2.attn.indexer.k_cache",
            vcfg_fp8.cache_config,
            compress_ratio=2,
        )
    spec_fp8 = idx_fp8.get_kv_cache_spec(vcfg_fp8)
    assert isinstance(spec_fp8, MLAAttentionSpec)
    assert spec_fp8.alignment == ROW_BYTES
    assert spec_fp8.page_size_padded is None

    vcfg = _make_dsv41_vllm_config(block_size=16, index_n_heads=4, index_topk=4)
    idx_rope_calls: list[tuple[int, list[int]]] = []

    def _spy_idx_rope(
        inputs: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        idx_rope_calls.append((inputs.ndim, positions.tolist()))
        return _identity_rope_op(inputs, positions, cos_sin_cache)

    monkeypatch.setattr(DeepseekV41Indexer, "kv_rope_op", staticmethod(_spy_idx_rope))

    with set_current_vllm_config(vcfg):
        idx_cache = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.19.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        unbound_op = TpuSparseAttnIndexer(idx_cache, topk_tokens=4)
        with pytest.raises(RuntimeError, match="callback is unbound"):
            unbound_op(
                torch.zeros((2, HIDDEN_SIZE), dtype=torch.bfloat16),
                torch.zeros((2, 4, INDEX_HEAD_DIM), dtype=torch.bfloat16),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.ones((2, 4, 1), dtype=torch.float32),
            )

        topk_buf = torch.full((8, 4), -9, dtype=torch.int32)
        cand_buf = torch.full((8, 4), -9, dtype=torch.int32)
        producer_idx = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=2,
            prefix="model.layers.19.attn.indexer",
            owns_k=True,
            k_cache=idx_cache,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=True,
        )
        consumer_idx = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=None,
            compress_ratio=1,
            prefix="model.layers.20.attn.indexer",
            owns_k=False,
            k_cache=idx_cache,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=False,
        )
    _init_module_bf16(producer_idx, 21)
    _init_module_bf16(consumer_idx, 22)

    q_dummy = torch.ones((2, 4, INDEX_HEAD_DIM), dtype=torch.bfloat16)
    w_dummy = torch.ones((2, 4, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires positions or a prior forward"):
        producer_idx.select_topk(q_dummy, w_dummy)

    rotary = _DummyRotaryEmb(max_pos=64)
    q_lora = torch.randn((2, Q_LORA_RANK), dtype=torch.bfloat16)
    idx_w = torch.ones((2, 4), dtype=torch.bfloat16)
    pos2 = torch.tensor([0, 1], dtype=torch.int32)
    with override_forward_context(_forward_ctx(None)):
        q_split, q_scale, scaled_w = producer_idx(
            q_lora,
            torch.ones((2, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            idx_w,
            pos2,
            rotary,
        )
        assert q_scale is None
        assert q_split is not None and scaled_w is not None
        out_empty = producer_idx.select_topk(q_split, scaled_w)
        assert out_empty.tolist() == [[-1] * 4, [-1] * 4]
        assert topk_buf[:2].tolist() == [[-1] * 4, [-1] * 4]
        cq, _, cw = consumer_idx(
            q_lora,
            torch.ones((2, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            idx_w,
            pos2,
            rotary,
        )
        assert cq is not None and cw is not None
        assert consumer_idx.select_topk(cq, cw).tolist() == [[-1] * 4, [-1] * 4]

    # Spy on single-bank dsv41_paged_indexer_topk to check 3D pool identity & offsets.
    calls: list[dict[str, Any]] = []

    def _spy_paged_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        k_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        topk: int,
        candidate_indices: torch.Tensor | None = None,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        del weights, seq_lens, cu_q_lens, compress_ratio
        calls.append(
            {
                "k_pool": k_pool,
                "block_tables": block_tables.clone(),
                "candidate_indices": None
                if candidate_indices is None
                else candidate_indices.clone(),
            }
        )
        return (
            torch.arange(topk, dtype=torch.int32, device=q.device)
            .unsqueeze(0)
            .expand(q.shape[0], topk)
            .contiguous()
        )

    monkeypatch.setattr(idx_mod.indexer_op, "dsv41_paged_indexer_topk", _spy_paged_topk)

    # Case A: Unit-scale 3D root pool (scale=1, records_per_page=32 -> page_rows=16).
    raw_unit = torch.zeros((4, 16, ROW_WORDS), dtype=torch.int32)
    idx_cache.bind_kv_cache(
        raw_unit,
        raw_cache=raw_unit,
        layout=((4, 32, INDEX_HEAD_DIM), torch.bfloat16, 16 * ROW_BYTES, 0),
    )
    meta_idx = _make_attn_meta(
        block_tables=torch.tensor([[0, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([40], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    idx_rope_calls.clear()
    with override_forward_context(
        _forward_ctx({"model.layers.19.attn.indexer.k_cache": meta_idx})
    ):
        q_s, _, w_s = producer_idx(
            q_lora,
            torch.ones((2, MAIN_HEAD_DIM), dtype=torch.bfloat16),
            idx_w,
            torch.tensor([38, 39], dtype=torch.int32),
            rotary,
        )
        producer_idx.indexer_op(
            torch.zeros((2, HIDDEN_SIZE), dtype=torch.bfloat16),
            (cast(torch.Tensor, q_s), None),
            None,
            cast(torch.Tensor, w_s),
        )
    # Ratio-2 key RoPE uses (pos // 2) * 2 ([38, 38]); query RoPE uses raw [38, 39].
    assert idx_rope_calls == [(2, [38, 38]), (3, [38, 39])]
    assert calls[-1]["k_pool"] is raw_unit
    assert calls[-1]["block_tables"].tolist() == [[-1, 2]]
    assert calls[-1]["candidate_indices"] is None
    assert cand_buf[1, 0].item() == 4

    # Case B: Folded dyadic 3D root pool (scale=4, page_offset=2, raw=(4, 64, 128)).
    raw_folded = torch.zeros((4, 64, ROW_WORDS), dtype=torch.int32)
    idx_cache.bind_kv_cache(
        raw_folded,
        raw_cache=raw_folded,
        layout=(
            (4, 32, INDEX_HEAD_DIM),
            torch.bfloat16,
            64 * ROW_BYTES,
            32 * ROW_BYTES,
        ),
    )
    with override_forward_context(
        _forward_ctx({"model.layers.19.attn.indexer.k_cache": meta_idx})
    ):
        consumer_idx.select_topk(
            cast(torch.Tensor, q_s),
            cast(torch.Tensor, w_s),
            torch.tensor([38, 39], dtype=torch.int32),
        )
    folded_pool = calls[-1]["k_pool"]
    assert folded_pool.is_contiguous()
    assert folded_pool.storage_offset() == 0
    assert folded_pool.shape == (4 * 4, 16, ROW_WORDS)
    assert calls[-1]["block_tables"].tolist() == [[-1, 2 * 4 + 2]]
    assert calls[-1]["candidate_indices"] is not None
    assert calls[-1]["candidate_indices"].shape == (2, 16)

    # Case C: Consumer skips candidate_indices when max_keys <= candidate window.
    dense_idx_pool = torch.zeros((3, 8, ROW_WORDS), dtype=torch.int32)
    idx_cache.kv_cache = dense_idx_pool
    meta_small = _make_attn_meta(
        block_tables=torch.tensor([[1]], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    with override_forward_context(
        _forward_ctx({"model.layers.19.attn.indexer.k_cache": meta_small})
    ):
        consumer_idx.select_topk(
            cast(torch.Tensor, q_s),
            cast(torch.Tensor, w_s),
            torch.tensor([6, 7], dtype=torch.int32),
        )
    assert calls[-1]["k_pool"] is dense_idx_pool
    assert calls[-1]["block_tables"].tolist() == [[1]]
    assert calls[-1]["candidate_indices"] is None


def test_should_compose_allocator_compressor_and_indexer_with_stubbed_ops(
    cpu_dist_ctx: None,
    cpu_cache_ops: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx, cpu_cache_ops
    monkeypatch.setattr(
        TpuPlatform, "_find_non_ssm_backend", lambda *_a: PallasAttentionBackend
    )
    monkeypatch.setattr(
        idx_mod.indexer_op, "dsv41_paged_indexer_topk", _ref_paged_indexer_topk
    )
    monkeypatch.setattr(
        DeepseekCompressor, "kv_rope_op", staticmethod(_identity_rope_op)
    )
    monkeypatch.setattr(
        DeepseekV41Indexer, "kv_rope_op", staticmethod(_identity_rope_op)
    )
    vcfg = _make_dsv41_vllm_config(
        block_size=16, max_model_len=256, index_n_heads=4, index_topk=4
    )

    with set_current_vllm_config(vcfg):
        idx_l2 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.2.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l8 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.8.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l14 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.14.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l20 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.20.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=1,
        )
        comp_l2 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=2,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.2.attn.compressor",
            k_cache_prefix="model.layers.2.attn",
        )
        comp_l20 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=1,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.20.attn.compressor",
            k_cache_prefix="model.layers.20.attn",
        )
        for lid, ratio in ((2, 2), (20, 1)):
            vcfg.compilation_config.static_forward_context[
                f"model.layers.{lid}.attn"
            ] = _DummyMlaCacheLayer(
                MLAAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=MAIN_HEAD_DIM,
                    dtype=torch.uint8,
                    tokens_per_state=ratio,
                    cache_dtype_str="fp8",
                    alignment=ROW_BYTES,
                    model_version="deepseek_v4",
                    state_content_bytes=640,
                )
            )
        cand_buf = torch.full((16, 4), -1, dtype=torch.int32)
        topk_buf = torch.full((16, 4), -1, dtype=torch.int32)
        indexer_l2 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=2,
            prefix="model.layers.2.attn.indexer",
            owns_k=True,
            k_cache=idx_l2,
            main_head_dim=MAIN_HEAD_DIM,
        )
        indexer_l19 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=2,
            prefix="model.layers.19.attn.indexer",
            owns_k=False,
            k_cache=idx_l2,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=True,
        )
        indexer_l20 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=1,
            prefix="model.layers.20.attn.indexer",
            owns_k=True,
            k_cache=idx_l20,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=False,
        )
    for seed_i, mod in enumerate(
        (comp_l2, comp_l20, indexer_l2, indexer_l19, indexer_l20), start=31
    ):
        _init_module_bf16(mod, seed_i)

    runner = _RunnerTestHarness(vcfg, torch.device("cpu"))
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 4)
    mgr.initialize_kv_cache(kv_cfg)

    assert idx_l2.paged_cache_rows is not None
    assert idx_l8.paged_cache_rows is not None
    assert idx_l14.paged_cache_rows is not None
    assert idx_l20.paged_cache_rows is not None
    # L2/L8/L20 pack into a 64-row dyadic pool while L14 stays unit-scale (16 rows).
    assert idx_l2.paged_cache_rows.page_stride > idx_l2.paged_cache_rows.shape[1]
    assert idx_l14.paged_cache_rows.page_stride == idx_l14.paged_cache_rows.shape[1]
    assert idx_l20.paged_cache_rows.raw.shape == (4, 64, ROW_WORDS)

    rotary = _DummyRotaryEmb(max_pos=128)
    num_tokens = 8
    positions = torch.arange(num_tokens, dtype=torch.int32)
    hidden = torch.randn((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16) * 0.2
    q_lora = torch.randn((num_tokens, Q_LORA_RANK), dtype=torch.bfloat16) * 0.2
    weights = torch.ones((num_tokens, 4), dtype=torch.bfloat16)

    shared_meta = _make_attn_meta(
        block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
    )
    meta_dict = {
        "model.layers.2.attn.compressor.state_cache": shared_meta,
        "model.layers.2.attn": shared_meta,
        "model.layers.2.attn.indexer.k_cache": shared_meta,
        "model.layers.20.attn": shared_meta,
        "model.layers.20.attn.indexer.k_cache": shared_meta,
    }
    with override_forward_context(_forward_ctx(meta_dict)):
        lat2 = comp_l2(comp_l2.project(hidden), positions)
        comp_l2.insert_cache(lat2, positions, rotary)
        q2, _, w2 = indexer_l2(q_lora, lat2, weights, positions, rotary)
        assert q2 is not None and w2 is not None
        topk_l2 = indexer_l2.select_topk(q2, w2)
        assert topk_l2.shape == (num_tokens, 4)
        assert sorted(topk_l2[7].tolist()) == [0, 1, 2, 3]

        q19, _, w19 = indexer_l19(q_lora, None, weights, positions, rotary)
        assert q19 is not None and w19 is not None
        indexer_l19.select_topk(q19, w19)
        assert cand_buf[7, 0].item() == 0

        lat20 = comp_l20(comp_l20.project(hidden), positions)
        comp_l20.insert_cache(lat20, positions, rotary)
        q20, _, w20 = indexer_l20(q_lora, lat20, weights, positions, rotary)
        assert q20 is not None and w20 is not None
        topk_l20 = indexer_l20.select_topk(q20, w20)
        assert topk_l20.shape == (num_tokens, 4)
        assert bool((topk_l20[7] >= 0).all())


def test_should_keep_null_block_zero_when_builder_clamps_ring_and_runner_zeroes(
    cpu_dist_ctx: None,
    cpu_cache_ops: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_dist_ctx
    monkeypatch.setattr(
        TpuPlatform, "_find_non_ssm_backend", lambda *_a: PallasAttentionBackend
    )
    vcfg = _make_dsv41_vllm_config(
        block_size=16, max_model_len=256, max_num_seqs=4, index_n_heads=4, index_topk=4
    )
    monkeypatch.setattr(
        DeepseekCompressor, "kv_rope_op", staticmethod(_identity_rope_op)
    )

    with set_current_vllm_config(vcfg):
        comp = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=2,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.2.attn.compressor",
            k_cache_prefix="model.layers.2.attn",
        )
        _init_module_bf16(comp, 41)
        vcfg.compilation_config.static_forward_context["model.layers.2.attn"] = (
            _DummyMlaCacheLayer(
                MLAAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=MAIN_HEAD_DIM,
                    dtype=torch.uint8,
                    tokens_per_state=2,
                    cache_dtype_str="fp8",
                    alignment=ROW_BYTES,
                    model_version="deepseek_v4",
                    state_content_bytes=640,
                )
            )
        )

    assert comp.state_cache is not None
    state_spec = comp.state_cache.get_kv_cache_spec(vcfg)
    assert isinstance(state_spec, CircularBufferSpec)

    runner = _RunnerTestHarness(vcfg, torch.device("cpu"), max_num_reqs=4)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 5)
    mgr.initialize_kv_cache(kv_cfg)

    state_gid = next(
        idx
        for idx, g in enumerate(kv_cfg.kv_cache_groups)
        if comp.state_cache.prefix in g.layer_names
    )
    cpu_bt = torch.tensor([[1], [2], [9], [9]], dtype=torch.int32)
    runner.input_batch.block_table[state_gid] = SimpleNamespace(
        max_num_blocks_per_req=1,
        get_cpu_tensor=lambda: cpu_bt,
    )
    runner.position_ids = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    seq_lens = torch.tensor([2, 1, 0, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)
    builder = AttentionMetadataBuilder(
        kv_cache_spec=kv_cfg.kv_cache_groups[state_gid].kv_cache_spec,
        layer_names=kv_cfg.kv_cache_groups[state_gid].layer_names,
        vllm_config=vcfg,
        device=torch.device("cpu"),
        runner=runner,
        kv_cache_group_id=state_gid,
    )
    builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
    )
    runner.set_builder_context(builder_ctx)
    common_meta = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=4,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=cpu_bt,
        slot_mapping=torch.zeros(8, dtype=torch.int64),
    )
    built_meta = builder.build(0, common_meta)
    assert built_meta.block_tables is not None
    assert built_meta.block_tables.tolist() == [1, 2, 0, 0]

    kv_score = torch.full((8, 2 * MAIN_HEAD_DIM), 3.25, dtype=torch.float32)
    with override_forward_context(_forward_ctx({comp.state_cache.prefix: built_meta})):
        comp(kv_score, runner.position_ids)

    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].abs().max().item()) == 0
    state_pcr = comp.state_cache.paged_cache_rows
    assert state_pcr is not None
    assert int(state_pcr.raw[1].abs().max().item()) > 0
    assert int(state_pcr.raw[2].abs().max().item()) > 0

    runner.zero_blocks_public([0, 1, 2])
    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].abs().max().item()) == 0
        assert int(raw_root[1].abs().max().item()) == 0
        assert int(raw_root[2].abs().max().item()) == 0

    # Direct save_compress_norm must also mask raw block-0 slots (< ring_capacity).
    ring_cap = state_pcr.shape[1]
    kv_block0_probe = torch.ones((4, 2 * MAIN_HEAD_DIM), dtype=torch.float32)
    kv_block0_probe[2, :MAIN_HEAD_DIM] = 5.0
    kv_block0_probe[2, MAIN_HEAD_DIM:] = 10.0
    kv_block0_probe[3, :MAIN_HEAD_DIM] = -1.0
    kv_block0_probe[3, MAIN_HEAD_DIM:] = -10.0
    block0_slots = torch.tensor([ring_cap, ring_cap + 1, 0, 1], dtype=torch.int32)
    block0_pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    direct_block0_out = save_compress_norm(
        kv_block0_probe,
        block0_pos,
        state_pcr,
        block0_slots,
        comp.norm.weight,
        comp.rms_norm_eps,
        2,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert int(state_pcr.raw[0].abs().max().item()) == 0
    expected_unpaired_r3 = rms_norm(
        kv_block0_probe[3:4, :MAIN_HEAD_DIM],
        comp.norm.weight.float(),
        comp.rms_norm_eps,
    ).to(torch.bfloat16)[0]
    assert torch.allclose(direct_block0_out[3], expected_unpaired_r3, atol=1e-2)

    cpu_cache_ops.clear()
    compiled_fn = torch.compile(
        lambda kvs, pos, sm: (
            save_compress_norm(
                kvs, pos, state_pcr, sm, comp.norm.weight, comp.rms_norm_eps, 2
            ),
            publish_candidate_blocks(
                torch.zeros((8, 4), dtype=torch.int32),
                pos,
                torch.tensor(3, dtype=torch.int32),
                2,
                4,
                4,
            ),
        ),
        backend="eager",
        fullgraph=True,
    )
    with torch.no_grad():
        lat_traced, cand_traced = compiled_fn(
            kv_score,
            runner.position_ids,
            torch.tensor([8, 9, 16, -1, -1, -1, -1, -1], dtype=torch.int32),
        )
    assert lat_traced.shape == (8, MAIN_HEAD_DIM)
    assert cand_traced.shape == (8, 4)
    assert cpu_cache_ops and all(cpu_cache_ops)


@pytest.mark.tpu_test
def test_tpu_should_compile_and_run_compressor_indexer_on_tpu7x(
    request: pytest.FixtureRequest,
) -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only test

    request.getfixturevalue("dist_ctx")
    dev = torch.device("tpu:0")
    vcfg = _make_dsv41_vllm_config(
        block_size=16,
        max_model_len=256,
        max_num_seqs=4,
        index_n_heads=4,
        index_topk=4,
        device="tpu",
    )
    with set_current_vllm_config(vcfg):
        idx_l2 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.2.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l8 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.8.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l14 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.14.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=2,
        )
        idx_l20 = IndexerCache(
            INDEX_HEAD_DIM,
            torch.bfloat16,
            "model.layers.20.attn.indexer.k_cache",
            vcfg.cache_config,
            compress_ratio=1,
        )
        comp_l2 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=2,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.2.attn.compressor",
            k_cache_prefix="model.layers.2.attn",
        )
        comp_l20 = DeepseekCompressor(
            vllm_config=vcfg,
            compress_ratio=1,
            hidden_size=HIDDEN_SIZE,
            head_dim=MAIN_HEAD_DIM,
            rotate=True,
            prefix="model.layers.20.attn.compressor",
            k_cache_prefix="model.layers.20.attn",
        )
        for lid, ratio in ((2, 2), (20, 1)):
            vcfg.compilation_config.static_forward_context[
                f"model.layers.{lid}.attn"
            ] = _DummyMlaCacheLayer(
                MLAAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=MAIN_HEAD_DIM,
                    dtype=torch.uint8,
                    tokens_per_state=ratio,
                    cache_dtype_str="fp8",
                    alignment=ROW_BYTES,
                    model_version="deepseek_v4",
                    state_content_bytes=640,
                )
            )
        cand_buf = torch.full((16, 4), -1, dtype=torch.int32, device=dev)
        topk_buf = torch.full((16, 4), -1, dtype=torch.int32, device=dev)
        indexer_l2 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=2,
            prefix="model.layers.2.attn.indexer",
            owns_k=True,
            k_cache=idx_l2,
            main_head_dim=MAIN_HEAD_DIM,
        )
        indexer_l19 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=2,
            prefix="model.layers.19.attn.indexer",
            owns_k=False,
            k_cache=idx_l2,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=True,
        )
        indexer_l20 = DeepseekV41Indexer(
            vllm_config=vcfg,
            config=vcfg.model_config.hf_config,
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            quant_config=None,
            cache_config=vcfg.cache_config,
            topk_indices_buffer=topk_buf,
            compress_ratio=1,
            prefix="model.layers.20.attn.indexer",
            owns_k=True,
            k_cache=idx_l20,
            main_head_dim=MAIN_HEAD_DIM,
            candidate_block_buffer=cand_buf,
            candidate_block_size=4,
            candidate_write=False,
        )
        rotary = upstream_attn.build_deepseek_v4_rope(
            vcfg.model_config.hf_config,
            head_dim=MAIN_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            max_position_embeddings=256,
            compress_ratio=1,
        )
    rotary = rotary.to(dev)
    for seed_i, mod in enumerate(
        (comp_l2, comp_l20, indexer_l2, indexer_l19, indexer_l20), start=51
    ):
        _init_module_bf16(mod, seed_i)
        mod.to(dev)

    runner = _RunnerTestHarness(vcfg, dev, max_num_reqs=4)
    mgr = KVCacheManager(runner)
    specs = mgr.get_kv_cache_spec()
    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 4)
    mgr.initialize_kv_cache(kv_cfg)

    assert idx_l2.paged_cache_rows is not None and idx_l8.paged_cache_rows is not None
    assert idx_l14.paged_cache_rows is not None and idx_l20.paged_cache_rows is not None
    assert idx_l2.paged_cache_rows.raw.device.type == "tpu"
    assert idx_l14.paged_cache_rows.page_stride == idx_l14.paged_cache_rows.shape[1]
    assert idx_l20.paged_cache_rows.raw.shape == (4, 64, ROW_WORDS)

    assert comp_l2.state_cache is not None
    state_gid = next(
        idx
        for idx, g in enumerate(kv_cfg.kv_cache_groups)
        if comp_l2.state_cache.prefix in g.layer_names
    )
    cpu_bt = torch.tensor([[1], [2], [9], [9]], dtype=torch.int32)
    runner.input_batch.block_table[state_gid] = SimpleNamespace(
        max_num_blocks_per_req=1,
        get_cpu_tensor=lambda: cpu_bt,
    )
    positions_tpu = torch.tensor(
        [0, 1, 2, 3, 0, 1, 0, 0], dtype=torch.int32, device=dev
    )
    runner.position_ids = positions_tpu
    seq_lens_tpu = torch.tensor([4, 2, 0, 0], dtype=torch.int32, device=dev)
    cu_q_cpu = torch.tensor([0, 4, 6, 6, 6], dtype=torch.int32)
    cu_q_tpu = cu_q_cpu.to(dev)
    builder = AttentionMetadataBuilder(
        kv_cache_spec=kv_cfg.kv_cache_groups[state_gid].kv_cache_spec,
        layer_names=kv_cfg.kv_cache_groups[state_gid].layer_names,
        vllm_config=vcfg,
        device=dev,
        runner=runner,
        kv_cache_group_id=state_gid,
    )
    builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=True,
        seq_lens=seq_lens_tpu,
        query_start_loc=cu_q_tpu,
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32, device=dev),
    )
    runner.set_builder_context(builder_ctx)
    common_meta = CommonAttentionMetadata(
        query_start_loc=cu_q_tpu,
        query_start_loc_cpu=cu_q_cpu,
        seq_lens=seq_lens_tpu,
        num_reqs=4,
        num_actual_tokens=6,
        max_query_len=4,
        max_seq_len=4,
        block_table_tensor=cpu_bt,
        slot_mapping=torch.zeros(8, dtype=torch.int64, device=dev),
    )
    state_meta_tpu = builder.build(0, common_meta)

    paged_meta_tpu = _make_attn_meta(
        block_tables=torch.tensor(
            [[1, 2], [2, 3], [0, 0], [0, 0]], dtype=torch.int32, device=dev
        ),
        seq_lens=seq_lens_tpu,
        query_start_loc=cu_q_tpu,
    )
    meta_map_tpu = {
        comp_l2.state_cache.prefix: state_meta_tpu,
        "model.layers.2.attn": paged_meta_tpu,
        "model.layers.2.attn.indexer.k_cache": paged_meta_tpu,
        "model.layers.20.attn": paged_meta_tpu,
        "model.layers.20.attn.indexer.k_cache": paged_meta_tpu,
    }

    gen = torch.Generator().manual_seed(99)
    hidden_tpu = (
        (torch.randn((8, HIDDEN_SIZE), generator=gen, dtype=torch.float32) * 0.25)
        .bfloat16()
        .to(dev)
    )
    q_lora_tpu = (
        (torch.randn((8, Q_LORA_RANK), generator=gen, dtype=torch.float32) * 0.25)
        .bfloat16()
        .to(dev)
    )
    weights_tpu = torch.ones((8, 4), dtype=torch.bfloat16, device=dev)

    def _compiled_step(
        h_in: torch.Tensor,
        qr_in: torch.Tensor,
        w_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lat2 = comp_l2(comp_l2.project(h_in), pos_in)
        comp_l2.insert_cache(lat2, pos_in, rotary)
        q2, _, w2 = indexer_l2(qr_in, lat2, w_in, pos_in, rotary)
        assert q2 is not None and w2 is not None
        tk2 = indexer_l2.select_topk(q2, w2)
        q19, _, w19 = indexer_l19(qr_in, None, w_in, pos_in, rotary)
        assert q19 is not None and w19 is not None
        indexer_l19.select_topk(q19, w19)
        lat20 = comp_l20(comp_l20.project(h_in), pos_in)
        comp_l20.insert_cache(lat20, pos_in, rotary)
        q20, _, w20 = indexer_l20(qr_in, lat20, w_in, pos_in, rotary)
        assert q20 is not None and w20 is not None
        tk20 = indexer_l20.select_topk(q20, w20)
        return lat2, tk2, tk20

    torch.compiler.reset()
    compiled_fn = torch.compile(
        _compiled_step, backend="tpu", fullgraph=True, dynamic=False
    )
    with (
        set_vllm_model_wrapper_context(mesh=cast(Any, None), vllm_config=vcfg),
        override_forward_context(_forward_ctx(meta_map_tpu)),
        torch.no_grad(),
    ):
        for mod in (comp_l2, comp_l20, indexer_l2, indexer_l19, indexer_l20):
            _ = mod.kv_rope_op
        lat2_out, tk2_out, tk20_out = compiled_fn(
            hidden_tpu, q_lora_tpu, weights_tpu, positions_tpu
        )

    assert torch.isfinite(lat2_out.cpu()).all()
    assert tk2_out.shape == (8, 4) and tk20_out.shape == (8, 4)
    assert sorted(tk2_out.cpu()[3, :2].tolist()) == [0, 1]
    assert tk2_out.cpu()[5, 0].item() == 0
    assert sorted(tk20_out.cpu()[3].tolist()) == [0, 1, 2, 3]
    assert cand_buf.cpu()[3, 0].item() == 0
    assert cand_buf.cpu()[5, 0].item() == 0

    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].cpu().abs().max().item()) == 0
    runner.zero_blocks_public([0, 1, 2])
    for raw_root in runner.kv_cache_raw_tensors:
        assert int(raw_root[0].cpu().abs().max().item()) == 0
        assert int(raw_root[1].cpu().abs().max().item()) == 0
        assert int(raw_root[2].cpu().abs().max().item()) == 0
