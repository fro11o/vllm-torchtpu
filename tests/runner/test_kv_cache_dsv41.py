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
"""CPU and TPU7x tests for DsV41KVCacheAllocator, KVCacheManager, and Raiden hooks."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.core import kv_cache_planning as kcu
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    dsv41_gather_sparse_fp8_kv,
    dsv41_paged_indexer_topk,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import DsV41CacheMixin
from vllm_torchtpu.models.vllm.deepseek_v41 import cache as cache_mod
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector
from vllm_torchtpu.offload.raiden_store import (
    has_multi_shapes_kv_caches,
    resolve_kernel_geometry,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kv_cache_dsv41 import (
    DsV41CacheLayer,
    DsV41KVCacheAllocator,
    is_cache_for_ds_v41,
)
from vllm_torchtpu.runner.kv_cache_manager import (
    KVCacheManager,
    check_kv_caches_cover_block_ids,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.utils import tpu_bind_kv_cache

GIB_BYTES: int = 1 << 30
SPARSECORE_CAP_BYTES: int = 16 << 30
MAX_L20_SINGLE_BANK_BLOCKS: int = 524_288


class _StubCacheModule(DsV41CacheMixin, AttentionLayerBase):
    def __init__(self, spec: KVCacheSpec | None) -> None:
        super().__init__()
        self.spec = spec

    def get_attn_backend(self) -> type[Any]:
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        del vllm_config
        return self.spec


class _DualDsV41AndV4SWACache(DsV41CacheMixin, DeepseekV4SWACache):
    def __init__(self, spec: KVCacheSpec) -> None:
        torch.nn.Module.__init__(self)
        self.spec = spec
        self.backend_cls = PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return self.spec


class _PureDsV4SWACache(DeepseekV4SWACache):
    def __init__(self, spec: KVCacheSpec) -> None:
        torch.nn.Module.__init__(self)
        self.spec = spec
        self.backend_cls = PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return self.spec


class _PlainAttnLayer(torch.nn.Module, AttentionLayerBase):
    def __init__(self, spec: KVCacheSpec) -> None:
        super().__init__()
        self.spec = spec

    def get_attn_backend(self) -> type[Any]:
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return self.spec


class _FakeTpuPool:
    def __init__(self, backing: torch.Tensor) -> None:
        self.backing = backing
        self.shape = tuple(backing.shape)
        self.device = SimpleNamespace(type="tpu")


class _ZeroBlockHarness(TPUModelRunner):
    def __init__(
        self,
        raw_tensors: list[Any],
        device: Any,
        max_num_reqs: int = 4,
        *,
        enable_dsv41_owner: bool = True,
        kv_cache_config: Any = None,
        pool_block_split: int = 1,
    ) -> None:
        self.kv_cache_raw_tensors = raw_tensors
        self._pool_block_split = pool_block_split
        self._mamba_copy_plan = []
        self.kv_cache_config = kv_cache_config
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.bm_pool_layout = None
        self.ds_v41_zero_owner = (
            DsV41KVCacheAllocator(self) if enable_dsv41_owner else None
        )
        self.recorded_copies: list[tuple[list[int], list[int]]] = []

    def _copy_mamba_state_blocks(
        self, raws: list[Any], src: torch.Tensor, dst: torch.Tensor
    ) -> None:
        src_list = src.tolist()
        dst_list = dst.tolist()
        self.recorded_copies.append((src_list, dst_list))
        for raw in raws:
            target = raw.backing if isinstance(raw, _FakeTpuPool) else raw
            for src_id, dst_id in zip(src_list, dst_list, strict=True):
                target[dst_id] = target[src_id].clone()

    def zero_blocks_public(self, block_ids: list[int]) -> None:
        self._zero_block_ids(block_ids)

    def apply_kv_cache_block_copies_public(self, copies: list[Any]) -> None:
        self._apply_kv_cache_block_copies(copies)


def _build_dsv41_specs(
    kv_sources: dict[int, int] | None = None,
    index_sources: dict[int, int] | None = None,
    num_swa_layers: int = 43,
    block_size: int = 16,
    ring_block_size: int = 8,
    head_dim: int = 512,
    index_head_dim: int = 128,
) -> dict[str, KVCacheSpec]:
    resolved_kv = {2: 2, 8: 2, 14: 2, 20: 1} if kv_sources is None else dict(kv_sources)
    resolved_idx = (
        {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}
        if index_sources is None
        else dict(index_sources)
    )
    init_mla_bs = math.lcm(block_size, *resolved_kv.values(), *resolved_idx.values())
    specs: dict[str, KVCacheSpec] = {}
    for layer_id, ratio in resolved_kv.items():
        specs[f"model.layers.{layer_id}.attn"] = MLAAttentionSpec(
            block_size=init_mla_bs,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            cache_dtype_str="fp8",
            alignment=512,
            model_version="deepseek_v4",
            state_content_bytes=(head_dim * 5) // 4,
        )
        if ratio > 1:
            specs[f"model.layers.{layer_id}.attn.compressor.state_cache"] = (
                CircularBufferSpec(
                    block_size=ring_block_size,
                    num_kv_heads=1,
                    head_size=head_dim * 2,
                    head_size_v=0,
                    dtype=torch.float32,
                )
            )
    for layer_id, ratio in resolved_idx.items():
        specs[f"model.layers.{layer_id}.attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=init_mla_bs,
            num_kv_heads=1,
            head_size=index_head_dim,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
            alignment=512,
        )
    for layer_id in range(num_swa_layers):
        specs[f"model.layers.{layer_id}.attn.swa_cache"] = SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_dim,
            dtype=torch.bfloat16,
            sliding_window=128,
            cache_dtype_str="fp8",
            alignment=512,
            model_version="deepseek_v4",
        )
    return specs


def _make_blhnc_vllm_config(block_size: int = 16) -> VllmConfig:
    vcfg = VllmConfig(
        cache_config=CacheConfig(block_size=block_size),
        device_config=DeviceConfig(device="cpu"),
    )
    vcfg.cache_config.kv_cache_layout = "BLHNC"
    return vcfg


def _make_manager_runner(
    vcfg: VllmConfig, forward_ctx: dict[str, Any]
) -> SimpleNamespace:
    vcfg.compilation_config.static_forward_context = forward_ctx

    def _may_reinit(kv_cache_config: KVCacheConfig, block_sizes: list[int]) -> None:
        del block_sizes
        tables = [
            SimpleNamespace(
                get_cpu_tensor=lambda: torch.zeros((4, 8), dtype=torch.int32)
            )
            for _ in kv_cache_config.kv_cache_groups
        ]
        runner.input_batch = SimpleNamespace(block_table=tables)

    precompile_seed_spy = MagicMock()
    runner = SimpleNamespace(
        vllm_config=vcfg,
        cache_config=vcfg.cache_config,
        model_config=SimpleNamespace(get_head_size=lambda: 512),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=vcfg.scheduler_config,
        speculative_config=None,
        device=torch.device("cpu"),
        kv_cache_dtype=torch.bfloat16,
        _unified_kv_layout=False,
        _is_async_drafter=False,
        _mamba_num_blocks=None,
        _uniform_mamba_layout=False,
        use_spmd=False,
        enforce_eager=True,
        max_num_reqs=4,
        kv_cache_config=None,
        kv_caches=[],
        kv_cache_raw_tensors=[],
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
        block_table_cpu=torch.zeros((4, 8), dtype=torch.int32),
        attn_groups=[],
        ds_v41_zero_owner=None,
        may_reinitialize_input_batch=_may_reinit,
        precompile_seed_spy=precompile_seed_spy,
        _precompile_dense_kv_cache_block_copies=MagicMock(),
        _initialize_attention_kernels=MagicMock(),
        _precompile_substitute_placeholder_token=MagicMock(),
        _precompile_mamba_state_seed_copies=precompile_seed_spy,
    )
    return runner


@pytest.fixture
def cpu_cache_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gather(pool: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        pool_2d = pool.reshape(-1, 128)
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
        del after, page_tokens
        out = bank.clone()
        sub = out.view(-1, updates.shape[1])
        loc = slots.long()
        valid = (loc >= 0) & (loc < sub.shape[0])
        sub[loc[valid]] = updates[valid]
        return out

    def _pack_fp8(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = latents[:, 0].to(torch.int32)
        nope = base.unsqueeze(1).expand(-1, 128).contiguous()
        rope = (base + 1000).unsqueeze(1).expand(-1, 32).contiguous()
        return nope, rope

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_gather_cache_rows", _gather)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _write)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_pack_fp8_kv_rows", _pack_fp8)


def test_should_keep_runner_imports_acyclic_and_base_layer_abstract() -> None:
    base_layer = DsV41CacheLayer()
    assert is_cache_for_ds_v41(base_layer) is True
    assert is_cache_for_ds_v41(object()) is False
    with pytest.raises(NotImplementedError):
        base_layer.bind_kv_cache(None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import vllm_torchtpu.runner.kv_cache_dsv41; "
                "import vllm_torchtpu.runner.kv_cache_manager; "
                "import vllm_torchtpu.runner.tpu_runner; "
                "assert not any("
                "m.startswith('vllm_torchtpu.models.vllm.deepseek_v41') "
                "for m in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("index_sources", "expected_layers", "expected_roots", "expected_bpb"),
    [
        ({2: 2, 8: 2, 14: 2, 20: 1}, 54, 8, 143_360),
        ({2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}, 58, 10, 208_896),
    ],
)
def test_should_enable_blhnc_and_multi_shape_raiden_offload_when_arch_is_dsv41(
    index_sources: dict[int, int],
    expected_layers: int,
    expected_roots: int,
    expected_bpb: int,
) -> None:
    model_cfg = MagicMock(spec=ModelConfig)
    model_cfg.tokenizer = "deepseek-ai/DeepSeek-V4"
    model_cfg.architectures = ["DeepseekV41ForCausalLM"]
    model_cfg.architecture = "DeepseekV41ForCausalLM"
    model_cfg.get_head_size.return_value = 512
    model_cfg.max_model_len = 2048
    model_cfg.is_multimodal_model = False
    model_cfg.is_hybrid = False
    model_cfg.is_attention_free = False
    model_cfg.use_mla = True
    model_cfg.hf_text_config = SimpleNamespace(model_type="deepseek_v4")

    with (
        patch.object(TpuPlatform, "uses_host_device_handling", return_value=True),
        patch("vllm_torchtpu.patch_registry.apply"),
        patch(
            "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
        ),
    ):
        vcfg = VllmConfig()
        vcfg.model_config = model_cfg
        vcfg.cache_config = CacheConfig()
        TpuPlatform.check_and_update_config(vcfg)
    assert vcfg.cache_config.kv_cache_layout == "BLHNC"

    other_vcfg = cast(
        VllmConfig,
        SimpleNamespace(model_config=SimpleNamespace(architecture="Qwen2ForCausalLM")),
    )
    assert has_multi_shapes_kv_caches(vcfg) is True
    assert has_multi_shapes_kv_caches(other_vcfg) is False

    raw_specs = _build_dsv41_specs(index_sources=index_sources)
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, 16)
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    upstream_bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    assert upstream_bpb == expected_bpb
    small_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 4)

    k_bs, per_block_shapes, kv_dtypes, dev_bs = resolve_kernel_geometry(vcfg, small_cfg)
    assert k_bs == 64 and dev_bs == 64
    assert kv_dtypes == {torch.uint8, torch.bfloat16}
    assert isinstance(per_block_shapes, set) and len(per_block_shapes) >= 2

    runner = _make_manager_runner(vcfg, {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(small_cfg, lambda n: prepared[n], kv_caches)
    assert len(kv_caches) == expected_layers

    captured: list[Any] = []
    fake_connector = SimpleNamespace(
        connector_worker=SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                num_blocks=small_cfg.num_blocks,
                kv_cache_groups=small_cfg.kv_cache_groups,
                hisparse_host_num_blocks=None,
                prefix_cacheable_group_ids=(0,),
            ),
            _init_worker=captured.append,
        )
    )
    TPURaidenOffloadingConnector.register_kv_caches(fake_connector, kv_caches)
    assert len(captured) == 1
    canonical = captured[0]
    assert len(canonical.tensors) == expected_roots
    assert len(canonical.group_data_refs) == 1
    assert sum(t.page_size_bytes for t in canonical.tensors) == expected_bpb

    for idx, pool in enumerate(runner.kv_cache_raw_tensors):
        pool[2].fill_(idx + 11)
    saved_block2 = [entry.tensor[2].clone() for entry in canonical.tensors]
    allocator.zero_blocks([2])
    for pool in runner.kv_cache_raw_tensors:
        assert int(pool[2].abs().max().item()) == 0
    for entry, host_bytes in zip(canonical.tensors, saved_block2, strict=True):
        entry.tensor[2].copy_(host_bytes)
    for idx, pool in enumerate(runner.kv_cache_raw_tensors):
        assert bool((pool[2] == idx + 11).all())


def test_should_prioritize_dsv41_before_dsv4_in_kv_cache_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_specs = _build_dsv41_specs(
        kv_sources={20: 1}, index_sources={20: 1}, num_swa_layers=1
    )
    vcfg = _make_blhnc_vllm_config(block_size=16)
    dual_swa = _DualDsV41AndV4SWACache(raw_specs["model.layers.0.attn.swa_cache"])
    forward_ctx: dict[str, Any] = {
        "model.layers.0.attn": _StubCacheModule(None),
        "model.layers.20.attn": _StubCacheModule(raw_specs["model.layers.20.attn"]),
        "model.layers.20.attn.indexer.k_cache": _StubCacheModule(
            raw_specs["model.layers.20.attn.indexer.k_cache"]
        ),
        "model.layers.0.attn.swa_cache": dual_swa,
    }
    runner = _make_manager_runner(vcfg, forward_ctx)
    manager = KVCacheManager(cast(TPUModelRunner, runner))

    with patch.object(
        TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
    ):
        specs = manager.get_kv_cache_spec()
    assert "model.layers.0.attn" not in specs
    assert specs["model.layers.20.attn"].block_size == 32

    groups = kcu.get_kv_cache_groups(vcfg, specs.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    kv_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 3)

    monkeypatch.setattr("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", 1)
    with pytest.raises(NotImplementedError, match="the DSv4.1 word-pool KV layout"):
        manager.initialize_kv_cache(kv_cfg)
    monkeypatch.setattr("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", 0)

    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        ),
        patch.object(manager.ds_v4, "_initialize_ds_v4_kv_cache") as ds_v4_spy,
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.check_kv_caches_cover_block_ids",
            wraps=check_kv_caches_cover_block_ids,
        ) as dsv41_cover_spy,
    ):
        manager.initialize_kv_cache(kv_cfg)
    ds_v4_spy.assert_not_called()
    dsv41_cover_spy.assert_called_once()
    assert runner.ds_v41_zero_owner is manager.ds_v41
    assert dual_swa.paged_cache_rows is not None

    v4_swa = _PureDsV4SWACache(raw_specs["model.layers.0.attn.swa_cache"])
    v4_ctx = {"model.layers.0.attn.swa_cache": v4_swa}
    v4_runner = _make_manager_runner(_make_blhnc_vllm_config(block_size=16), v4_ctx)
    v4_mgr = KVCacheManager(cast(TPUModelRunner, v4_runner))
    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        ),
        patch.object(v4_mgr.ds_v4, "_initialize_ds_v4_kv_cache") as v4_init_spy,
        patch.object(v4_mgr.ds_v41, "initialize_kv_cache") as v41_init_spy,
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.check_kv_caches_cover_block_ids",
            wraps=check_kv_caches_cover_block_ids,
        ) as v4_cover_spy,
    ):
        v4_specs = v4_mgr.get_kv_cache_spec()
        v4_groups = kcu.get_kv_cache_groups(v4_runner.vllm_config, v4_specs.copy())
        v4_cfg = kcu.get_kv_cache_config_from_groups(
            v4_runner.vllm_config, v4_groups, 1024 * 1024
        )
        v4_mgr.initialize_kv_cache(v4_cfg)
    v4_init_spy.assert_called_once()
    v41_init_spy.assert_not_called()
    v4_cover_spy.assert_not_called()
    assert v4_runner.ds_v41_zero_owner is None

    monkeypatch.setattr("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", 1)
    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        ),
        pytest.raises(NotImplementedError, match="DSv4 packed KV layout"),
    ):
        v4_mgr.initialize_kv_cache(v4_cfg)
    monkeypatch.setattr("vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV", 0)

    plain_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    plain_layer = _PlainAttnLayer(plain_spec)
    plain_layer.kv_cache = [torch.empty(0)]
    plain_ctx = {"model.layers.0.attn": plain_layer}
    plain_runner = _make_manager_runner(_make_blhnc_vllm_config(16), plain_ctx)
    plain_mgr = KVCacheManager(cast(TPUModelRunner, plain_runner))
    plain_groups = kcu.get_kv_cache_groups(
        plain_runner.vllm_config, {"model.layers.0.attn": plain_spec}
    )
    plain_cfg = kcu.get_kv_cache_config_from_groups(
        plain_runner.vllm_config, plain_groups, 1024 * 1024
    )
    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        ),
        patch.object(plain_mgr.ds_v4, "_initialize_ds_v4_kv_cache") as plain_v4_spy,
        patch.object(plain_mgr.ds_v41, "initialize_kv_cache") as plain_v41_spy,
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
            return_value=False,
        ),
    ):
        plain_mgr.initialize_kv_cache(plain_cfg)
    plain_v4_spy.assert_not_called()
    plain_v41_spy.assert_not_called()
    assert plain_runner.ds_v41_zero_owner is None
    assert len(plain_runner.kv_caches) == 1


@pytest.mark.parametrize(
    ("index_sources", "budget_gib", "expected_bpb", "expected_rows"),
    [
        (
            {2: 2, 8: 2, 14: 2, 20: 1},
            4,
            143_360,
            [64, 64, 32, 64, 16, 16, 16, 8],
        ),
        (
            {2: 2, 8: 2, 14: 2, 20: 1},
            70,
            143_360,
            [64, 64, 32, 64, 16, 16, 16, 8],
        ),
        (
            {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1},
            4,
            208_896,
            [64, 64, 32, 64, 64, 64, 16, 16, 16, 8],
        ),
        (
            {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1},
            102,
            208_896,
            [64, 64, 32, 64, 64, 64, 16, 16, 16, 8],
        ),
    ],
)
def test_should_pack_dsv41_into_contiguous_pools_with_zero_waste_up_to_sc_limit(
    index_sources: dict[int, int],
    budget_gib: int,
    expected_bpb: int,
    expected_rows: list[int],
) -> None:
    vcfg = _make_blhnc_vllm_config(block_size=16)
    raw_specs = _build_dsv41_specs(index_sources=index_sources)
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, 16)
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    full_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, budget_gib * GIB_BYTES)

    upstream_bpb = full_cfg.kv_cache_tensors[0].block_stride
    assert upstream_bpb == expected_bpb
    assert full_cfg.num_blocks == (budget_gib * GIB_BYTES) // upstream_bpb
    assert full_cfg.num_blocks * 64 * 512 <= SPARSECORE_CAP_BYTES

    small_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 6)
    runner = _make_manager_runner(vcfg, {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(small_cfg, lambda n: prepared[n], kv_caches)
    check_kv_caches_cover_block_ids(
        kv_caches,
        small_cfg.kv_cache_tensors,
        small_cfg.num_blocks,
        lambda n: prepared[n],
    )

    physical_bpb = sum(int(t.shape[1]) * 512 for t in runner.kv_cache_raw_tensors)
    assert physical_bpb == upstream_bpb
    assert [int(t.shape[1]) for t in runner.kv_cache_raw_tensors] == expected_rows

    for pool in runner.kv_cache_raw_tensors:
        assert pool.is_contiguous()
        assert pool.storage_offset() == 0
        assert pool.shape[0] == small_cfg.num_blocks
        assert pool.untyped_storage().nbytes() % small_cfg.num_blocks == 0

    for entry in kv_caches.values():
        tensors = entry if isinstance(entry, tuple) else (entry,)
        for tensor in tensors:
            assert tensor.stride(0) > 0
            assert tensor.storage_offset() == 0
            assert tensor.shape[0] == small_cfg.num_blocks


@pytest.mark.parametrize(
    ("index_sources", "expected_bpb", "expected_roots"),
    [
        ({2: 2, 8: 2, 14: 2, 20: 1}, 143_360, 8),
        ({2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}, 208_896, 10),
    ],
)
def test_should_accept_524288_blocks_and_reject_524289_single_bank_sc_limit(
    monkeypatch: pytest.MonkeyPatch,
    index_sources: dict[int, int],
    expected_bpb: int,
    expected_roots: int,
) -> None:
    vcfg = _make_blhnc_vllm_config(block_size=16)
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(
        _build_dsv41_specs(index_sources=index_sources), 16
    )
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    legal_cfg = kcu.get_kv_cache_config_from_groups(
        vcfg, groups, MAX_L20_SINGLE_BANK_BLOCKS * expected_bpb
    )
    assert legal_cfg.num_blocks == MAX_L20_SINGLE_BANK_BLOCKS
    over_cfg = dataclasses.replace(legal_cfg, num_blocks=MAX_L20_SINGLE_BANK_BLOCKS + 1)

    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_dsv41.synchronize_tensors", lambda *_a: None
    )
    with FakeTensorMode():
        runner = _make_manager_runner(vcfg, {})
        allocator = DsV41KVCacheAllocator(runner)
        kv_caches: dict[str, Any] = {}
        allocator.initialize_kv_cache(legal_cfg, lambda n: prepared[n], kv_caches)
        assert len(runner.kv_cache_raw_tensors) == expected_roots
        l20_nope = kv_caches["model.layers.20.attn"][0]
        assert l20_nope.shape == (MAX_L20_SINGLE_BANK_BLOCKS, 64, 128)
        assert l20_nope.shape[0] * l20_nope.shape[1] == 33_554_432
        assert l20_nope.nbytes == SPARSECORE_CAP_BYTES
        assert (
            sum(int(t.shape[1]) * 512 for t in runner.kv_cache_raw_tensors)
            == expected_bpb
        )

        with pytest.raises(ValueError, match="16 GiB single-bank SparseCore"):
            allocator.initialize_kv_cache(over_cfg, lambda n: prepared[n], {})


def test_should_keep_shared_pools_disjoint_and_preserve_cow_and_zero_blocks_when_bound(
    cpu_cache_ops: None,
) -> None:
    del cpu_cache_ops
    vcfg = _make_blhnc_vllm_config(block_size=16)
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(
        _build_dsv41_specs(num_swa_layers=12), 16
    )
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    upstream_bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 5)

    forward_ctx: dict[str, Any] = {
        name: _StubCacheModule(spec) for name, spec in prepared.items()
    }
    forward_ctx["shared.l2.alias"] = _StubCacheModule(None)
    runner = _make_manager_runner(vcfg, forward_ctx)
    runner.shared_kv_cache_layers = {"shared.l2.alias": "model.layers.2.attn"}
    runner.enforce_eager = False
    cfg.kv_cache_groups[0].layer_names.append("shared.l2.alias")

    allocator = DsV41KVCacheAllocator(runner)
    allocator.bind_paged_caches(forward_ctx)

    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(cfg, lambda n: prepared[n], kv_caches)
    runner.precompile_seed_spy.assert_called_once()
    tpu_bind_kv_cache(kv_caches, forward_ctx, runner.kv_caches)
    allocator.bind_paged_caches(forward_ctx)

    l2_pcr = forward_ctx["model.layers.2.attn"].paged_cache_rows
    l8_pcr = forward_ctx["model.layers.8.attn"].paged_cache_rows
    l20_pcr = forward_ctx["model.layers.20.attn"].paged_cache_rows
    alias_pcr = forward_ctx["shared.l2.alias"].paged_cache_rows
    assert l2_pcr is not None and l8_pcr is not None and l20_pcr is not None
    assert alias_pcr is not None and alias_pcr.raw is l2_pcr.raw
    assert l2_pcr.raw is l8_pcr.raw
    assert l2_pcr.rope is l8_pcr.rope

    bt = torch.tensor([[0, 1, 3]], dtype=torch.int32)
    l2_pages = l2_pcr.pages(bt)
    l8_pages = l8_pcr.pages(bt)
    assert l2_pages is not None and l2_pages.tolist() == [[-1, 2, 6]]
    assert l8_pages is not None and l8_pages.tolist() == [[-1, 3, 7]]

    l2_pcr.write(
        torch.tensor([2 * 32 + 5], dtype=torch.int32),
        torch.full((1, 512), 17.0, dtype=torch.bfloat16),
    )
    l8_pcr.write(
        torch.tensor([2 * 32 + 5], dtype=torch.int32),
        torch.full((1, 512), 29.0, dtype=torch.bfloat16),
    )
    assert int(l2_pcr.raw[2, 5, 0].item()) == 17
    assert int(l8_pcr.raw[2, 32 + 5, 0].item()) == 29
    assert int(l2_pcr.rope[2, 1, 32].item()) == 1017
    assert int(l8_pcr.rope[2, 8 + 1, 32].item()) == 1029

    swa_group = next(
        g for g in cfg.kv_cache_groups if g.layer_names[0].endswith(".swa_cache")
    )
    swa0_pcr = forward_ctx[swa_group.layer_names[0]].paged_cache_rows
    swa1_pcr = forward_ctx[swa_group.layer_names[1]].paged_cache_rows
    assert swa0_pcr is not None and swa1_pcr is not None
    swa0_pcr.write(
        torch.tensor([1 * 16 + 3], dtype=torch.int32),
        torch.full((1, 512), 4.5, dtype=torch.bfloat16),
    )
    swa1_pcr.write(
        torch.tensor([1 * 16 + 3], dtype=torch.int32),
        torch.full((1, 512), -7.25, dtype=torch.bfloat16),
    )
    assert float(
        swa0_pcr[torch.tensor([1]), torch.tensor([3])][0, 0].item()
    ) == pytest.approx(4.5)
    assert float(
        swa1_pcr[torch.tensor([1]), torch.tensor([3])][0, 0].item()
    ) == pytest.approx(-7.25)

    zero_harness = _ZeroBlockHarness(
        runner.kv_cache_raw_tensors, torch.device("cpu"), kv_cache_config=cfg
    )
    assert zero_harness.is_unified_pool_used() is True
    zero_harness.apply_kv_cache_block_copies_public(
        [SimpleNamespace(src_block_id=2, dst_block_id=4)]
    )
    assert int(l2_pcr.raw[4, 5, 0].item()) == 17
    assert int(l8_pcr.raw[4, 32 + 5, 0].item()) == 29

    runner.kv_cache_raw_tensors[0][1].fill_(999)
    zero_harness.zero_blocks_public([-1, 0, 2, 99])
    assert int(runner.kv_cache_raw_tensors[0][0].abs().max().item()) == 0
    assert int(runner.kv_cache_raw_tensors[0][1, 0, 0].item()) == 999
    assert int(l2_pcr.raw[2, 5, 0].item()) == 0
    assert int(l8_pcr.raw[2, 32 + 5, 0].item()) == 0
    assert int(l2_pcr.raw[4, 5, 0].item()) == 17

    split_pool = torch.zeros((5, 4, 128), dtype=torch.int32)
    split_pool[1:].fill_(88)
    split_harness = _ZeroBlockHarness(
        [split_pool], torch.device("cpu"), pool_block_split=2
    )
    split_harness.zero_blocks_public([0, 1, 2])
    assert int(split_pool[1, 0, 0].item()) == 88
    assert int(split_pool[2].abs().max().item()) == 0
    assert int(split_pool[3].abs().max().item()) == 0
    assert int(split_pool[4].abs().max().item()) == 0

    tpu_backing = torch.zeros((40, 8, 128), dtype=torch.int32)
    tpu_backing[1:].fill_(777)
    fake_tpu_pool = _FakeTpuPool(tpu_backing)
    tpu_zero = _ZeroBlockHarness(
        [fake_tpu_pool],
        torch.device("cpu"),
        max_num_reqs=4,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()] * 6),
    )
    targets = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    tpu_zero.zero_blocks_public([0, *targets])
    assert len(tpu_zero.recorded_copies) == 1
    assert tpu_zero.recorded_copies[0] == ([0] * 32, [*targets, *([0] * 22)])
    for cleared_id in targets:
        assert int(tpu_backing[cleared_id].abs().max().item()) == 0
    assert int(tpu_backing[1, 0, 0].item()) == 777


def test_should_leave_non_dsv41_unified_pool_untouched_when_owner_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcfg = SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=16,
        is_moe_model=False,
        enable_expert_parallel=False,
    )
    mcfg = SimpleNamespace(
        architecture="LlamaForCausalLM",
        runner_type="generate",
        enforce_eager=True,
        quantization=None,
        get_sliding_window=lambda: None,
        get_num_layers_by_block_type=lambda _pc, _bt: 1,
        get_num_kv_heads=lambda _pc: 1,
        get_head_size=lambda: 64,
        get_vocab_size=lambda: 128,
    )
    vllm_cfg = SimpleNamespace(
        parallel_config=pcfg,
        speculative_config=None,
        kv_transfer_config=None,
        quant_config=None,
        load_config=SimpleNamespace(),
        profiler_config=SimpleNamespace(torch_profiler_dir=""),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            cudagraph_mode=None,
            compile_sizes=[16],
            static_forward_context={},
        ),
        model_config=mcfg,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    def _stub_gpu_init(self: Any, cfg: SimpleNamespace, dev: torch.device) -> None:
        self.vllm_config = cfg
        self.model_config = cfg.model_config
        self.cache_config = cfg.cache_config
        self.parallel_config = cfg.parallel_config
        self.scheduler_config = cfg.scheduler_config
        self.load_config = cfg.load_config
        self.device = dev
        self.dtype = torch.bfloat16
        self.max_model_len = 64
        self.max_num_reqs = 4
        self.max_num_tokens = 16
        self.block_size = 16
        self.uses_mrope = False
        self.supports_mm_inputs = False
        self.sample_from_logits = None
        self.requests = {}
        self.mesh = None
        self.drafter = None
        self.lookback_token_ids = None
        self.input_batch = SimpleNamespace(block_table=[], _tpu_spec_room_guard=True)

    monkeypatch.setattr(GPUModelRunner, "__init__", _stub_gpu_init)
    for attr in (
        "_validate_libtpu_version",
        "RunnerStageTiming",
        "get_tpu_quantization_config",
        "validate_online_fp8",
        "install_kernel_block_tables",
    ):
        monkeypatch.setattr(f"vllm_torchtpu.runner.tpu_runner.{attr}", lambda *_a: None)
    for cm in (
        "_torch_tpu_wrapper",
        "set_vllm_model_wrapper_context",
        "set_current_vllm_config",
    ):
        monkeypatch.setattr(
            f"vllm_torchtpu.runner.tpu_runner.{cm}",
            lambda *_a, **_kw: contextlib.nullcontext(),
        )
    for method in ("_update_num_xla_graphs", "_create_mesh_for_parallelism"):
        monkeypatch.setattr(TPUModelRunner, method, lambda *_a: None)
    fresh_runner = TPUModelRunner(
        cast(VllmConfig, vllm_cfg),
        torch.device("cpu"),
        profiler_rank=0,
        profiler_world_size=1,
    )
    assert fresh_runner.ds_v41_zero_owner is None

    mamba_pool = torch.full((8, 4, 128), 321, dtype=torch.int32)
    mamba_pool[0].zero_()
    non_dsv41_runner = _ZeroBlockHarness(
        [mamba_pool], torch.device("cpu"), enable_dsv41_owner=False
    )
    assert non_dsv41_runner.is_unified_pool_used() is True

    non_dsv41_runner.zero_blocks_public([1, 2, 3])

    assert non_dsv41_runner.recorded_copies == []
    assert int(mamba_pool[1, 0, 0].item()) == 321
    assert int(mamba_pool[2, 0, 0].item()) == 321


def test_should_fail_fast_when_dsv41_null_block_zero_invariant_is_violated() -> None:
    corrupted_pool = torch.zeros((8, 4, 128), dtype=torch.int32)
    corrupted_pool[0, 0, 0] = 999
    corrupted_pool[2].fill_(55)
    harness = _ZeroBlockHarness([corrupted_pool], torch.device("cpu"))

    with pytest.raises(RuntimeError, match="null_block=0 must remain all-zero"):
        harness.zero_blocks_public([2])

    tpu_backing = torch.zeros((8, 4, 128), dtype=torch.int32)
    fake_tpu_pool = _FakeTpuPool(tpu_backing)
    tpu_harness = _ZeroBlockHarness([fake_tpu_pool], torch.device("cpu"))
    assert tpu_harness.ds_v41_zero_owner is not None
    with pytest.raises(ValueError, match="copy_blocks_fn is required"):
        tpu_harness.ds_v41_zero_owner.zero_blocks([2])


def test_should_reject_overlapping_same_group_bindings_when_corrupted() -> None:
    swa_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="fp8",
        alignment=512,
        model_version="deepseek_v4",
    )
    group_spec = UniformTypeKVCacheSpecs.from_specs(
        {"l0.swa_cache": swa_spec, "l1.swa_cache": swa_spec}
    )
    assert group_spec is not None
    cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["l0.swa_cache", "l1.swa_cache"], group_spec)
        ],
    )
    runner = _make_manager_runner(_make_blhnc_vllm_config(), {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(cfg, lambda _n: swa_spec, kv_caches)

    bad_group = KVCacheGroupSpec(["l0.swa_cache", "l0.swa_cache"], group_spec)
    corrupted_cfg = dataclasses.replace(cfg, kv_cache_groups=[bad_group])
    with pytest.raises(ValueError, match="DeepSeek-V4.1 KV overlay collision"):
        allocator.initialize_kv_cache(corrupted_cfg, lambda _n: swa_spec, {})


def test_should_reject_undersized_cache_and_zero_standalone_mixin_banks() -> None:
    ring_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=1024,
        head_size_v=0,
        dtype=torch.float32,
    )
    tensor_desc = KVCacheTensor(
        size=131072,
        layers=["l0.state"],
        layer_stride=32768,
        block_stride=32768,
        offset=0,
    )
    with pytest.raises(ValueError, match="Missing KV cache for attention layer"):
        check_kv_caches_cover_block_ids({}, [tensor_desc], 4, lambda _n: ring_spec)

    undersized_pair = (
        torch.zeros((4, 32, 128), dtype=torch.int32),
        torch.zeros((3, 8, 128), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="holds 3 blocks"):
        check_kv_caches_cover_block_ids(
            {"l0.state": undersized_pair},
            [tensor_desc],
            4,
            lambda _n: ring_spec,
        )

    zero_harness = _ZeroBlockHarness(
        [torch.ones((4, 8, 128), dtype=torch.int32)], torch.device("cpu")
    )
    zero_harness.zero_blocks_public([])
    zero_harness.zero_blocks_public([0, -1, 99])
    assert int(zero_harness.kv_cache_raw_tensors[0][1, 0, 0].item()) == 1


@pytest.mark.parametrize(
    ("kv_src", "idx_src", "swa_bs", "ring_bs", "head_dim", "idx_dim", "budget_gib"),
    [
        (None, None, 16, 16, 512, 128, 64),
        (None, None, 16, 32, 512, 128, 64),
        (None, None, 16, 128, 512, 128, 64),
        (None, None, 64, 8, 512, 128, 64),
        ({20: 1}, {20: 1}, 32, 8, 512, 128, 28),
        ({2: 4, 8: 3, 20: 1}, {2: 4, 8: 3, 20: 1}, 16, 8, 512, 128, 44),
        ({2: 1, 8: 1, 14: 1, 20: 1}, {2: 1, 8: 1, 14: 1, 20: 1}, 16, 8, 512, 128, 64),
        (None, None, 16, 8, 1024, 256, 64),
    ],
)
def test_should_keep_exact_budget_across_ring_capacities_block_sizes_and_head_dims(
    monkeypatch: pytest.MonkeyPatch,
    kv_src: dict[int, int] | None,
    idx_src: dict[int, int] | None,
    swa_bs: int,
    ring_bs: int,
    head_dim: int,
    idx_dim: int,
    budget_gib: int,
) -> None:
    vcfg = _make_blhnc_vllm_config(block_size=swa_bs)
    raw_specs = _build_dsv41_specs(
        kv_sources=kv_src,
        index_sources=idx_src,
        block_size=swa_bs,
        ring_block_size=ring_bs,
        head_dim=head_dim,
        index_head_dim=idx_dim,
    )
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, swa_bs)
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    full_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, budget_gib * GIB_BYTES)
    upstream_bpb = full_cfg.kv_cache_tensors[0].block_stride

    scale_blocks = 4
    scaled_sc_cap = (SPARSECORE_CAP_BYTES // full_cfg.num_blocks) * scale_blocks
    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_dsv41.SPARSECORE_MAX_TENSOR_BYTES",
        scaled_sc_cap,
    )
    small_cfg = kcu.get_kv_cache_config_from_groups(
        vcfg, groups, upstream_bpb * scale_blocks
    )
    runner = _make_manager_runner(vcfg, {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(small_cfg, lambda n: prepared[n], kv_caches)
    check_kv_caches_cover_block_ids(
        kv_caches,
        small_cfg.kv_cache_tensors,
        small_cfg.num_blocks,
        lambda n: prepared[n],
    )

    physical_bpb = sum(int(t.shape[1]) * 512 for t in runner.kv_cache_raw_tensors)
    assert physical_bpb == upstream_bpb
    for name in (f"model.layers.{lid}.attn" for lid in (kv_src or {2: 2, 20: 1})):
        entry = kv_caches[name]
        assert isinstance(entry, tuple) and len(entry) == 2
        for tensor in entry:
            scaled_bytes = (
                full_cfg.num_blocks * int(tensor.shape[1]) * int(tensor.shape[2]) * 4
            )
            assert scaled_bytes <= SPARSECORE_CAP_BYTES


def test_should_handle_edge_topologies_overflow_and_invalid_specs() -> None:
    swa_only = _build_dsv41_specs(kv_sources={}, index_sources={}, num_swa_layers=2)
    assert DsV41KVCacheAllocator.prepare_kv_cache_specs(swa_only, 16) == swa_only

    odd_state_spec = CircularBufferSpec(
        block_size=1, num_kv_heads=1, head_size=128, head_size_v=0, dtype=torch.float32
    )
    bad_full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    passthrough_specs = _build_dsv41_specs(
        kv_sources={20: 1},
        index_sources={20: 1},
        num_swa_layers=1,
        block_size=16,
        head_dim=576,
    )
    passthrough_specs["odd.compressor.state_cache"] = odd_state_spec
    passthrough_specs["extra.full"] = bad_full
    prep_passthrough = DsV41KVCacheAllocator.prepare_kv_cache_specs(
        passthrough_specs, 16
    )
    assert prep_passthrough["extra.full"] == bad_full

    overflow_specs = _build_dsv41_specs(
        kv_sources={2: 2},
        index_sources={2: 2},
        num_swa_layers=2,
        block_size=128,
        ring_block_size=32,
        head_dim=576,
    )
    prep_overflow = DsV41KVCacheAllocator.prepare_kv_cache_specs(overflow_specs, 128)
    assert prep_overflow["model.layers.2.attn"].block_size == 256

    runner = _make_manager_runner(_make_blhnc_vllm_config(), {})
    allocator = DsV41KVCacheAllocator(runner)
    empty_cfg = KVCacheConfig(num_blocks=2, kv_cache_tensors=[], kv_cache_groups=[])
    allocator.initialize_kv_cache(empty_cfg, lambda n: overflow_specs[n], {})
    assert runner.kv_cache_raw_tensors == []

    bad_group = UniformTypeKVCacheSpecs.from_specs({"bad.attn": bad_full})
    assert bad_group is not None
    bad_cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["bad.attn"], bad_group)],
    )
    with pytest.raises(ValueError, match="Unsupported DeepSeek-V4.1 KV cache layer"):
        allocator.initialize_kv_cache(bad_cfg, lambda _n: bad_full, {})

    swa_small = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="fp8",
        alignment=512,
        model_version="deepseek_v4",
    )
    swa_large = dataclasses.replace(swa_small, block_size=32)
    st_small = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32
    )
    st_large = dataclasses.replace(st_small, block_size=16)
    het_specs: dict[str, KVCacheSpec] = {
        "swa0.swa_cache": swa_small,
        "swa1.swa_cache": swa_large,
        "st_odd.compressor.state_cache": odd_state_spec,
        "st0.compressor.state_cache": st_small,
        "st1.compressor.state_cache": st_large,
    }
    het_cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["swa0.swa_cache"], swa_small),
            KVCacheGroupSpec(["swa1.swa_cache"], swa_large),
            KVCacheGroupSpec(["st_odd.compressor.state_cache"], odd_state_spec),
            KVCacheGroupSpec(["st0.compressor.state_cache"], st_small),
            KVCacheGroupSpec(["st1.compressor.state_cache"], st_large),
        ],
    )
    runner.shared_kv_cache_layers = {"unrelated.shared": "unrelated.target"}
    het_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(het_cfg, lambda n: het_specs[n], het_caches)
    assert len(het_caches) == 5
    allocator.bind_paged_caches({"unrelated.other": object()})


def test_should_split_non_pow2_mla_layers_into_dyadic_pools_for_swa_overlay() -> None:
    vcfg = _make_blhnc_vllm_config(block_size=16)
    base_specs = _build_dsv41_specs(
        kv_sources={2: 2, 6: 2, 10: 2},
        index_sources={},
        num_swa_layers=1,
        block_size=16,
        head_dim=256,
    )
    raw_specs = {k: v for k, v in base_specs.items() if "compressor" not in k}
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, 16)
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    upstream_bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    small_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 4)
    runner = _make_manager_runner(vcfg, {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(small_cfg, lambda n: prepared[n], kv_caches)

    pool_rows = sorted(
        (int(t.shape[1]) for t in runner.kv_cache_raw_tensors), reverse=True
    )
    assert 48 not in pool_rows
    assert all(r & (r - 1) == 0 for r in pool_rows)
    assert pool_rows == [32, 16, 8, 4]
    physical_bpb = sum(int(t.shape[1]) * 512 for t in runner.kv_cache_raw_tensors)
    assert physical_bpb == upstream_bpb


def test_should_prevent_unbudgeted_hbm_when_sc_bin_cap_clamps_mla_merging() -> None:
    vcfg = _make_blhnc_vllm_config(block_size=16)
    raw_specs = _build_dsv41_specs(
        kv_sources={2: 2, 8: 2, 14: 2},
        index_sources={2: 2, 8: 2, 14: 2},
        num_swa_layers=43,
        block_size=16,
        ring_block_size=8,
        head_dim=512,
        index_head_dim=128,
    )
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, 16)
    assert prepared["model.layers.2.attn"].block_size == 128

    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    upstream_bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    small_cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 4)
    runner = _make_manager_runner(vcfg, {})
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(small_cfg, lambda n: prepared[n], kv_caches)
    physical_bpb = sum(int(t.shape[1]) * 512 for t in runner.kv_cache_raw_tensors)
    assert physical_bpb == upstream_bpb

    over_cfg = dataclasses.replace(small_cfg, num_blocks=MAX_L20_SINGLE_BANK_BLOCKS + 7)
    runner_over = _make_manager_runner(vcfg, {})
    allocator_over = DsV41KVCacheAllocator(runner_over)
    with (
        FakeTensorMode(),
        pytest.raises(ValueError, match="16 GiB single-bank SparseCore limit"),
    ):
        allocator_over.initialize_kv_cache(over_cfg, lambda n: prepared[n], {})


def test_should_align_shared_mla_and_rope_page_scales_for_csa_gather(
    cpu_cache_ops: None,
) -> None:
    del cpu_cache_ops
    vcfg = _make_blhnc_vllm_config(block_size=16)
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(
        _build_dsv41_specs(num_swa_layers=2), 16
    )
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, bpb * 4)
    forward_ctx = {name: _StubCacheModule(spec) for name, spec in prepared.items()}
    runner = _make_manager_runner(vcfg, forward_ctx)
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(cfg, lambda n: prepared[n], kv_caches)
    tpu_bind_kv_cache(kv_caches, forward_ctx, runner.kv_caches)
    allocator.bind_paged_caches(forward_ctx)

    l2_pcr = forward_ctx["model.layers.2.attn"].paged_cache_rows
    l8_pcr = forward_ctx["model.layers.8.attn"].paged_cache_rows
    l20_pcr = forward_ctx["model.layers.20.attn"].paged_cache_rows
    assert l2_pcr is not None and l8_pcr is not None and l20_pcr is not None
    assert l20_pcr.raw.is_contiguous() and l20_pcr.raw.storage_offset() == 0
    assert l20_pcr.raw.shape == (4, 64, 128)
    assert l20_pcr.rope is not None and l20_pcr.rope.is_contiguous()
    assert l20_pcr.rope.storage_offset() == 0 and l20_pcr.rope.shape == (4, 16, 128)

    slots_b1 = torch.arange(32, 36, dtype=torch.int32)
    l2_pcr.write(slots_b1, torch.full((4, 512), 41.0, dtype=torch.bfloat16))
    l8_pcr.write(slots_b1, torch.full((4, 512), 83.0, dtype=torch.bfloat16))

    bt_torch = torch.tensor([[1]], dtype=torch.int32)
    l2_bt = l2_pcr.pages(bt_torch)
    l8_bt = l8_pcr.pages(bt_torch)
    assert l2_bt is not None and l8_bt is not None

    nope_pages = l2_pcr.raw.view(-1, 32, 128)
    rope_pages = cast(torch.Tensor, l2_pcr.rope).view(-1, 8, 128)
    l2_phys, l8_phys = int(l2_bt[0, 0].item()), int(l8_bt[0, 0].item())
    assert l2_phys == 2 and l8_phys == 3
    assert int(nope_pages[l2_phys, 0, 0].item()) == 41
    assert int(rope_pages[l2_phys, 0, 0].item()) == 1041
    assert int(nope_pages[l8_phys, 0, 0].item()) == 83
    assert int(rope_pages[l8_phys, 0, 0].item()) == 1083


@pytest.mark.tpu_test
def test_should_roundtrip_3d_shared_mla_pools_on_tpu7x() -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only test

    dev = torch.device("tpu:0")
    vcfg = _make_blhnc_vllm_config(block_size=16)
    raw_specs = _build_dsv41_specs(
        kv_sources={2: 2, 8: 2, 20: 1},
        index_sources={2: 2, 8: 2, 20: 1},
        num_swa_layers=2,
    )
    prepared = DsV41KVCacheAllocator.prepare_kv_cache_specs(raw_specs, 16)
    groups = kcu.get_kv_cache_groups(vcfg, prepared.copy())
    upstream_bpb = (
        kcu.get_kv_cache_config_from_groups(vcfg, groups, GIB_BYTES)
        .kv_cache_tensors[0]
        .block_stride
    )
    cfg = kcu.get_kv_cache_config_from_groups(vcfg, groups, upstream_bpb * 4)

    forward_ctx = {name: _StubCacheModule(spec) for name, spec in prepared.items()}
    runner = _make_manager_runner(vcfg, forward_ctx)
    runner.device = dev
    allocator = DsV41KVCacheAllocator(runner)
    kv_caches: dict[str, Any] = {}
    allocator.initialize_kv_cache(cfg, lambda n: prepared[n], kv_caches)
    tpu_bind_kv_cache(kv_caches, forward_ctx, runner.kv_caches)
    allocator.bind_paged_caches(forward_ctx)

    l2_pcr = forward_ctx["model.layers.2.attn"].paged_cache_rows
    l8_pcr = forward_ctx["model.layers.8.attn"].paged_cache_rows
    idx2_pcr = forward_ctx["model.layers.2.attn.indexer.k_cache"].paged_cache_rows
    assert l2_pcr is not None and l8_pcr is not None and idx2_pcr is not None
    assert l2_pcr.raw is l8_pcr.raw

    gen = torch.Generator().manual_seed(1234)
    lat_l2 = (
        torch.randn((8, 512), generator=gen, dtype=torch.float32).bfloat16().to(dev)
    )
    lat_l8 = (
        (torch.randn((8, 512), generator=gen, dtype=torch.float32) * 2.0 + 0.5)
        .bfloat16()
        .to(dev)
    )
    slots_b1 = torch.arange(32, 40, dtype=torch.int32, device=dev)
    l2_pcr.write(slots_b1, lat_l2)
    l8_pcr.write(slots_b1, lat_l8)

    bt = torch.tensor([[1, 2]], dtype=torch.int32, device=dev)
    topk_idx = torch.arange(8, dtype=torch.int32, device=dev).unsqueeze(0)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)

    l2_bt = l2_pcr.pages(bt)
    l8_bt = l8_pcr.pages(bt)
    assert l2_bt is not None and l8_bt is not None
    gnope_l2, grope_l2 = dsv41_gather_sparse_fp8_kv(
        l2_pcr.raw.view(-1, 32, 128),
        cast(torch.Tensor, l2_pcr.rope).view(-1, 8, 128),
        topk_idx,
        l2_bt,
        cu_q,
        32,
    )
    gnope_l8, grope_l8 = dsv41_gather_sparse_fp8_kv(
        l8_pcr.raw.view(-1, 32, 128),
        cast(torch.Tensor, l8_pcr.rope).view(-1, 8, 128),
        topk_idx,
        l8_bt,
        cu_q,
        32,
    )
    assert not torch.equal(gnope_l2.cpu(), gnope_l8.cpu())
    assert torch.equal(gnope_l2[0].cpu(), l2_pcr.raw[1, :8].cpu())
    assert torch.equal(gnope_l8[0].cpu(), l8_pcr.raw[1, 32:40].cpu())
    assert grope_l2.shape == (1, 2, 128) and grope_l8.shape == (1, 2, 128)

    idx_keys = (
        torch.randn((8, 128), generator=gen, dtype=torch.float32).bfloat16().to(dev)
    )
    idx2_pcr.write(slots_b1, idx_keys)
    q_idx = torch.ones((1, 32, 128), dtype=torch.bfloat16, device=dev) * 0.1
    w_idx = torch.ones((1, 32, 1), dtype=torch.float32, device=dev)
    idx_bt = torch.tensor([[1 * 3 + 2, -1]], dtype=torch.int32, device=dev)
    topk_out = dsv41_paged_indexer_topk(
        q_idx,
        w_idx,
        idx2_pcr.raw.view(-1, 16, 128),
        idx_bt,
        torch.tensor([16], dtype=torch.int32, device=dev),
        cu_q,
        4,
        compress_ratio=2,
    )
    assert topk_out.shape == (1, 4)
    assert bool((topk_out.cpu() >= 0).all())
