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
"""Spec tests for bounded-replay tracking, SWA retention, and lookback staging."""

import contextlib
import inspect
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.config import CacheConfig
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu import (
    _patch_dflash_sliding_window_retained_tokens,
    patch_registry,
    utils,
)
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.runner.tpu_runner import (
    _LOOKBACK_RING_DEPTH,
    TPUModelRunner,
)


class _DummyLookbackModel(
    torch.nn.Module, utils.TokenLookbackModel, utils.RpaSmemPageTableModel
):
    def __init__(self, *, depth: int = 3, uses_smem: bool = False) -> None:
        super().__init__()
        self.token_lookback_depth = depth
        self.uses_rpa_smem_page_table = uses_smem
        self.last_kwargs: dict[str, object] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, inputs_embeds
        self.last_kwargs = kwargs
        return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)


class _DummyPlainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: dict[str, object] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, inputs_embeds
        self.last_kwargs = kwargs
        return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)


def _swa_kv_config() -> KVCacheConfig:
    swa_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=32,
        bounded_replay=True,
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=swa_spec)
        ],
    )


def _init_cpu_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_num_reqs: int = 16,
    max_num_tokens: int = 64,
    max_model_len: int = 64,
) -> TPUModelRunner:
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
            compile_sizes=[min(16, max_num_tokens), max_num_tokens],
            static_forward_context={},
        ),
        model_config=mcfg,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    def _stub_gpu_init(self: GPUModelRunner, cfg: SimpleNamespace, dev: object) -> None:
        self.vllm_config = cfg
        self.model_config = cfg.model_config
        self.cache_config = cfg.cache_config
        self.parallel_config = cfg.parallel_config
        self.scheduler_config = cfg.scheduler_config
        self.load_config = cfg.load_config
        self.lora_config = None
        self.device = dev
        self.dtype = torch.bfloat16
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
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
    return TPUModelRunner(
        vllm_cfg, torch.device("cpu"), profiler_rank=0, profiler_world_size=1
    )


def test_should_expose_model_mixins_and_drop_cpu_snapshots_in_pytree() -> None:
    assert not issubclass(utils.TokenLookbackModel, utils.RpaSmemPageTableModel)
    assert utils.RpaSmemPageTableModel.uses_rpa_smem_page_table is True
    assert utils.TokenLookbackModel.token_lookback_depth == 0
    assert _LOOKBACK_RING_DEPTH == 4

    bt = SimpleNamespace(
        max_num_blocks_per_req=1,
        blocks_per_kv_block=1,
        get_cpu_tensor=lambda: torch.tensor([[3], [5], [7], [9]], dtype=torch.int32),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        block_size=8,
        max_num_reqs=4,
        most_model_len=32,
        position_ids=torch.arange(4, dtype=torch.int32),
        _block_table_stage_cpu=None,
        _free_h2d_sources={},
        _registered_h2d_sources=set(),
        _unfenced_h2d_sources=[],
        attn_groups=[],
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt]),
    )
    spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        page_size_padded=2048,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["state.0"],
        vllm_config=MagicMock(),
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(  # noqa: SLF001 -- test runner builder context
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        query_start_loc_cpu=(0, 2, 4, 4, 4),
        seq_lens_cpu=(24, 16, 0, 0),
        prompt_lens_cpu=(20, 16, 0, 0),
        replay_starts_cpu=(8, 0, 0, 0),
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert (
        meta.query_start_loc_cpu,
        meta.seq_lens_cpu,
        meta.prompt_lens_cpu,
        meta.replay_starts_cpu,
    ) == ((0, 2, 4, 4, 4), (24, 16, 0, 0), (20, 16, 0, 0), (8, 0, 0, 0))
    leaves, treedef = jax.tree_util.tree_flatten(meta)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (
        restored.query_start_loc_cpu,
        restored.seq_lens_cpu,
        restored.prompt_lens_cpu,
        restored.replay_starts_cpu,
    ) == (None, None, None, None)


def test_should_track_replay_starts_and_gather_lookback_on_cold_and_warm_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _init_cpu_runner(monkeypatch)
    assert runner._replay_starts == {}  # noqa: SLF001 -- verify initial state
    assert runner._uses_rpa_smem_page_table is True  # noqa: SLF001 -- verify initial state
    assert runner._prefix_replay_tokens() == 0  # noqa: SLF001 -- verify initial state

    sched_out = SimpleNamespace(
        finished_req_ids={"finished_req"},
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="new_replayed",
                replay_start=16,
                num_computed_tokens=16,
                sampling_params=None,
            ),
            SimpleNamespace(
                req_id="new_plain",
                replay_start=0,
                num_computed_tokens=0,
                sampling_params=None,
            ),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["resumed_req", "active_req", "resumed_zero"],
            resumed_req_ids={"resumed_req", "resumed_zero"},
            num_computed_tokens=[24, 32, 0],
        ),
    )
    runner._replay_starts = None  # noqa: SLF001 -- verify None replay map is no-op
    with patch.object(GPUModelRunner, "_update_states", return_value=False):
        assert runner._update_states(sched_out) is False  # noqa: SLF001 -- direct unit test of state hook
    assert runner._replay_starts is None  # noqa: SLF001 -- verify uninitialized state preserved
    assert runner._build_replay_starts_cpu(0, 1, 2) == (0, 0)  # noqa: SLF001 -- None replay map yields zero tuple

    runner._replay_starts = {}  # noqa: SLF001 -- restore active replay map before KV config
    with patch.object(GPUModelRunner, "_update_states", return_value=False):
        assert runner._update_states(sched_out) is False  # noqa: SLF001 -- no-op when kv_cache_config is None
    assert runner._replay_starts == {}  # noqa: SLF001 -- no starts recorded without bounded replay

    runner.kv_cache_config = _swa_kv_config()
    assert runner._prefix_replay_tokens() == 32  # noqa: SLF001 -- verify replay window query
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True  # noqa: SLF001 -- populate replay starts via _update_states
    assert runner._replay_starts == {  # noqa: SLF001 -- verify recorded starts
        "new_replayed": 16,
        "resumed_req": 24,
    }

    runner._replay_starts = {  # noqa: SLF001 -- seed pre-eviction replay state
        "finished_req": 16,
        "resumed_req": 8,
        "survivor": 12,
    }
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True  # noqa: SLF001 -- direct unit test of eviction hook
    assert runner._replay_starts == {  # noqa: SLF001 -- verify finished evicted and survivor kept
        "survivor": 12,
        "new_replayed": 16,
        "resumed_req": 24,
    }

    buf_np = np.zeros((4, 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    runner.input_batch = SimpleNamespace(
        _tpu_spec_room_guard=True,
        num_reqs=3,
        req_ids=["req0", "req1", "req_cold"],
        num_computed_tokens_cpu=np.array([5, 6, 2, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 4, 4, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 7, 2, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [0, 42, 99, 77, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    runner.model = _DummyLookbackModel(depth=3, uses_smem=True)
    runner._replay_starts = {}  # noqa: SLF001 -- cold admission without SWA replay
    assert (runner._prepare_lookback_token_ids(0) == -1).all()  # noqa: SLF001 -- zero-request padding check
    runner.forward_model(
        torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)
    )
    out_prompt_only = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(out_prompt_only, torch.Tensor)
    assert out_prompt_only[0].tolist() == [-1, 13, 12]
    assert out_prompt_only[1].tolist() == [-1, -1, 23]
    # Cold partial lookback preserves valid token ID 0 while padding OOB slots with -1.
    assert out_prompt_only[2].tolist() == [42, 0, -1]
    assert out_prompt_only[3].tolist() == [-1, -1, -1]

    # Warm SWA bounded replay starting at token 6 for req1 exposes history up to
    # num_tokens_no_spec while req0 stays prompt-bounded.
    warm_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req1",
                replay_start=6,
                num_computed_tokens=6,
                sampling_params=None,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(warm_sched)  # noqa: SLF001 -- record warm replay start via _update_states
    runner.input_batch.num_reqs = 2
    runner.forward_model(
        torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    )
    out_replayed = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(out_replayed, torch.Tensor)
    assert out_replayed[0].tolist() == [-1, 13, 12]
    assert out_replayed[1].tolist() == [25, 24, 23]


def test_should_reflect_speculative_rejection_rollback_in_subsequent_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _init_cpu_runner(monkeypatch, max_num_reqs=2, max_num_tokens=16)
    runner.kv_cache_config = _swa_kv_config()
    buf_np = np.zeros((2, 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    runner.model = _DummyLookbackModel(depth=3, uses_smem=True)
    runner.input_batch = SimpleNamespace(
        _tpu_spec_room_guard=True,
        num_reqs=1,
        req_ids=["req0"],
        num_computed_tokens_cpu=np.array([7, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [100, 101, 102, 103, 200, 301, 302, 303],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    resume_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids={"req0"},
            num_computed_tokens=[7],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(resume_sched)  # noqa: SLF001 -- record resume start=7 via real _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    on_resume = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(on_resume, torch.Tensor)
    assert on_resume[0].tolist() == [302, 301, 200]

    # Without a new preemption/resume transition, _replay_starts stays at 7; a step
    # whose num_computed_tokens rolled back to 6 (or advanced to 8) masks generated
    # tokens >= num_prompt_tokens (4) to -1 rather than reading past the prompt.
    runner.input_batch.token_ids_cpu[0, 5:7] = [301, 401]
    runner.input_batch.num_computed_tokens_cpu[0] = 6
    runner.input_batch.num_tokens_no_spec[0] = 7
    continue_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids=set(),
            num_computed_tokens=[6],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(continue_sched)  # noqa: SLF001 -- advance non-resumed state via real _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    non_resume_rollback = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(non_resume_rollback, torch.Tensor)
    assert non_resume_rollback[0].tolist() == [-1, -1, 103]

    runner.input_batch.num_computed_tokens_cpu[0] = 8
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    later_decode = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(later_decode, torch.Tensor)
    assert later_decode[0].tolist() == [-1, -1, -1]

    # When preempted and re-resumed at rolled-back computed count 6, _update_states
    # refreshes replay start to 6 so lookback exposes bonus history [301, 200, 103].
    runner.input_batch.num_computed_tokens_cpu[0] = 6
    reresume_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req0"},
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids={"req0"},
            num_computed_tokens=[6],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(reresume_sched)  # noqa: SLF001 -- re-resume at rolled-back computed=6 via _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    after_reresume = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(after_reresume, torch.Tensor)
    assert after_reresume[0].tolist() == [301, 200, 103]


def test_should_scale_lookback_staging_independent_of_max_model_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_runner = _init_cpu_runner(monkeypatch, max_num_reqs=4, max_model_len=64)
    large_runner = _init_cpu_runner(monkeypatch, max_num_reqs=4, max_model_len=131072)
    outputs = []
    for runner in (small_runner, large_runner):
        buf_np = np.zeros((4, 3), dtype=np.int32)
        runner.lookback_token_ids = SimpleNamespace(
            np=buf_np, copy_to_gpu=lambda b=buf_np: torch.from_numpy(b.copy())
        )
        runner._replay_starts = {}  # noqa: SLF001 -- zero replay state
        runner.input_batch = SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens_cpu=np.array([4, 3, 0, 0], dtype=np.int32),
            num_prompt_tokens=np.array([6, 6, 0, 0], dtype=np.int32),
            num_tokens_no_spec=np.array([6, 6, 0, 0], dtype=np.int32),
            token_ids_cpu=np.array(
                [
                    [10, 20, 30, 40, 50, 60],
                    [70, 80, 90, 100, 110, 120],
                ],
                dtype=np.int32,
            ),
        )
        outputs.append(runner._prepare_lookback_token_ids(2))  # noqa: SLF001 -- stage fixed chunk window

    assert outputs[0].shape == outputs[1].shape == (4, 3)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0][:2].tolist() == [[40, 30, 20], [90, 80, 70]]


def test_should_fence_four_slot_lookback_ring_across_wraparound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _init_cpu_runner(monkeypatch)
    runner.scheduler_config.async_scheduling = True
    runner._replay_starts = {}  # noqa: SLF001 -- zero replay state
    runner.input_batch = SimpleNamespace(
        req_ids=["req0"],
        num_computed_tokens_cpu=np.array([3, 0, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([8, 0, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 0, 0, 0], dtype=np.int32),
        token_ids_cpu=np.zeros((4, 8), dtype=np.int32),
    )

    cpu_arrays = [np.zeros((4, 2), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)]
    runner._lookback_staging_ring = tuple(  # noqa: SLF001 -- stub ring buffers
        SimpleNamespace(np=arr, copy_to_gpu=lambda a=arr: torch.from_numpy(a.copy()))
        for arr in cpu_arrays
    )
    runner._lookback_staging_fences = None  # noqa: SLF001 -- exercise lazy fence allocation
    runner._lookback_staging_idx = 0  # noqa: SLF001 -- reset ring cursor
    runner.input_batch.token_ids_cpu[0, :3] = [11, 12, 13]
    out_cpu = runner._prepare_lookback_token_ids(1)  # noqa: SLF001 -- stage on CPU async path
    assert out_cpu[0].tolist() == [13, 12]
    assert runner._lookback_staging_fences == [None] * _LOOKBACK_RING_DEPTH  # noqa: SLF001 -- no TPU events on CPU

    runner.device = SimpleNamespace(type="tpu")
    ring_arrays = [
        np.zeros((4, 2), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)
    ]
    events_log: list[tuple[str, int, list[int]]] = []

    class _FakeTpuEvent:
        slot_cursor = 0

        def __init__(self) -> None:
            self.slot = _FakeTpuEvent.slot_cursor % _LOOKBACK_RING_DEPTH
            _FakeTpuEvent.slot_cursor += 1
            self.recorded_row0: list[int] = []

        def record(self) -> None:
            self.recorded_row0 = ring_arrays[self.slot][0].tolist()
            events_log.append(("record", self.slot, list(self.recorded_row0)))

        def synchronize(self) -> None:
            current_row0 = ring_arrays[self.slot][0].tolist()
            assert current_row0 == self.recorded_row0
            events_log.append(("sync", self.slot, current_row0))

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=_FakeTpuEvent), raising=False
    )
    runner._lookback_staging_ring = tuple(  # noqa: SLF001 -- stub TPU ring buffers
        SimpleNamespace(
            np=arr,
            copy_to_gpu=lambda slot=i: SimpleNamespace(
                device=SimpleNamespace(type="tpu"), slot=slot
            ),
        )
        for i, arr in enumerate(ring_arrays)
    )
    runner._lookback_staging_fences = [None] * _LOOKBACK_RING_DEPTH  # noqa: SLF001 -- reset fences
    runner._lookback_staging_idx = 0  # noqa: SLF001 -- reset cursor

    for step in range(6):
        base = (step + 1) * 100
        runner.input_batch.token_ids_cpu[0, :3] = [base + 1, base + 2, base + 3]
        out = runner._prepare_lookback_token_ids(1)  # noqa: SLF001 -- step async ring
        slot = step % _LOOKBACK_RING_DEPTH
        assert out.slot == slot
        assert ring_arrays[slot][0].tolist() == [base + 3, base + 2]
        if step >= 1:
            prev_slot = (step - 1) % _LOOKBACK_RING_DEPTH
            prev_base = step * 100
            assert ring_arrays[prev_slot][0].tolist() == [prev_base + 3, prev_base + 2]

    assert [entry[:2] for entry in events_log] == [
        ("record", 0),
        ("record", 1),
        ("record", 2),
        ("record", 3),
        ("sync", 0),
        ("record", 0),
        ("sync", 1),
        ("record", 1),
    ]


@pytest.mark.parametrize(
    (
        "is_lookback",
        "depth",
        "uses_smem",
        "most_len",
        "max_num_tokens",
        "expected_caps",
    ),
    [
        (True, 3, False, 64, 64, (16, 16)),
        (True, 3, False, 64, 10, (10, 10)),
        (True, 0, True, 64, 64, (4, 6)),
        (False, 0, True, None, 64, (4, None)),
    ],
)
def test_should_configure_lookback_ring_and_smem_caps_on_model_load(
    is_lookback: bool,
    depth: int,
    uses_smem: bool,
    most_len: int | None,
    max_num_tokens: int,
    expected_caps: tuple[int, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _init_cpu_runner(
        monkeypatch, max_num_reqs=16, max_num_tokens=max_num_tokens
    )
    runner.most_model_len = most_len
    runner.max_num_blocks_per_req = 8
    runner._pp_is_last = False  # noqa: SLF001 -- disable greedy top-k setup
    runner._attention_capacity = {"cached": (1, 1, 1)}  # noqa: SLF001 -- verify cache clear on refresh
    runner._decode_device_cache_key = ("cached",)  # noqa: SLF001 -- verify cache clear on refresh
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        block_table=[SimpleNamespace(max_num_blocks_per_req=8)] if is_lookback else [],
    )
    runner.block_table_cpu = torch.zeros((16, 8), dtype=torch.int32)
    runner._make_buffer = lambda rows, cols, *, dtype: SimpleNamespace(  # noqa: SLF001 -- stub device staging buffer
        shape=(rows, cols), dtype=dtype
    )
    runner._prepare_lookback_token_ids = MagicMock(  # noqa: SLF001 -- isolate forward_model hook
        return_value=torch.full((16, max(1, depth)), 7, dtype=torch.int32)
    )
    runner._initialize_pallas_kernels = lambda: None  # noqa: SLF001 -- stub kernel compile
    runner.kv_cache_manager = SimpleNamespace(get_kv_cache_spec=lambda: {})
    model = (
        _DummyLookbackModel(depth=depth, uses_smem=uses_smem)
        if is_lookback
        else _DummyPlainModel()
    )
    orig_device = torch.device
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.torch.device",
        lambda dev: orig_device("cpu") if dev == "tpu" else orig_device(dev),
    )
    for flag in (
        "TPU_ROPE_CACHE_TRUNCATE",
        "TPU_ROPE_CACHE_ROW_MAJOR",
        "TPU_MOE_HASH_TABLE_ROW_MAJOR",
        "FUSED_MOE_DECODE_KERNEL",
        "TPU_MOE_SKIP_PADDED_TOKENS",
        "USE_MOE_TP_DECODE_KERNEL",
    ):
        monkeypatch.setattr(f"vllm_torchtpu.runner.tpu_runner.envs.{flag}", False)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.get_model_loader",
        lambda _cfg: SimpleNamespace(load_model=lambda **_kw: model),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_max_num_seqs_for_pages",
        lambda pages: 4 if pages == 8 else 6,
    )
    monkeypatch.setattr(
        GPUModelRunner, "may_reinitialize_input_batch", lambda *_a: None
    )

    runner.load_model()
    assert runner.get_kv_cache_spec() == {}
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps
    runner.may_reinitialize_input_batch(None, [16])  # type: ignore[arg-type] -- None KVCacheConfig exercises pre-init fallback
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps
    out, aux = runner.forward_model(
        torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    )
    assert out.shape == (2, 4) and aux is None
    assert (runner.lookback_token_ids is not None) is (depth > 0)
    assert (
        runner._lookback_staging_ring is not None  # noqa: SLF001 -- verify 4-slot ring creation
        and len(runner._lookback_staging_ring) == _LOOKBACK_RING_DEPTH  # noqa: SLF001 -- verify 4-slot ring length
    ) is (depth > 0)
    assert ("lookback_token_ids" in model.last_kwargs) is (depth > 0)


def test_should_slice_chunked_lookback_via_chunk_span_in_forward_model() -> None:
    runner = _make_runner(
        num_computed_tokens=[2, 3, 6],
        prompt_tokens=[4, 4, 4],
        scheduled_tokens=[1, 1, 1],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=4,
    )
    runner.kv_cache_config = _swa_kv_config()
    runner._prepare_lookback_token_ids = MethodType(  # noqa: SLF001 -- bind real lookback preparer
        TPUModelRunner._prepare_lookback_token_ids,  # noqa: SLF001 -- bind real lookback preparer
        runner,
    )
    runner._lookback_staging_ring = None  # noqa: SLF001 -- sync staging mode
    buf_np = np.zeros((4, 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    # Asymmetric num_tokens_no_spec across rows [2, 3, 7] discriminates batch_row=2
    # from chunk-local row=1 when preparing chunk start_index=1 with replay on req2.
    runner.input_batch.num_tokens_no_spec = np.array([2, 3, 7], dtype=np.int32)
    runner._replay_starts = {"req0": 55, "req2": 77}  # noqa: SLF001 -- distinct starts on rows 0 and 2

    TPUModelRunner._prepare_inputs(runner, _scheduler_output([1, 1, 1]), 1, 3, 3)  # noqa: SLF001 -- prepare second chunk at start_index=1
    assert runner._lookback_chunk_span == (1, 2)  # noqa: SLF001 -- verify non-zero start_index span
    assert runner._attn_metadata_builder_ctx.replay_starts_cpu == (0, 77, 0, 0)  # noqa: SLF001 -- verify start_index offset in replay_starts_cpu

    # Activate replay match on chunk row 1 (batch_row=2, "req2") at computed=6.
    runner._replay_starts = {"req0": 55, "req2": 6}  # noqa: SLF001 -- match req2 computed=6 on chunk row 1
    runner.model = _DummyLookbackModel(depth=3, uses_smem=True)
    TPUModelRunner.forward_model(
        runner,
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    lb = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(lb, torch.Tensor)
    assert lb.shape == (4, 3)
    assert lb[0].tolist() == [202, 201, 200]
    assert lb[1].tolist() == [305, 304, 303]
    assert lb[2].tolist() == [-1, -1, -1]
    assert lb[3].tolist() == [-1, -1, -1]

    # When _lookback_chunk_span is None, forward_model slices full [0..num_reqs).
    runner._lookback_chunk_span = None  # noqa: SLF001 -- verify None span fallback
    TPUModelRunner.forward_model(
        runner,
        torch.zeros(3, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
    )
    lb_full = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(lb_full, torch.Tensor)
    assert lb_full[0].tolist() == [101, 100, -1]
    assert lb_full[1].tolist() == [202, 201, 200]
    assert lb_full[2].tolist() == [305, 304, 303]


@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("dp_size_and_lockstep", [(1, False), (8, True)])
@pytest.mark.parametrize("profiling", [False, True])
def test_should_populate_cpu_scheduling_snapshots_across_layouts_and_dummy_modes(
    pcp_size: int,
    dp_size_and_lockstep: tuple[int, bool],
    profiling: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_size, dp_lockstep = dp_size_and_lockstep
    if pcp_size > 1:
        pcp_mod = "vllm_torchtpu.layers.core.pcp_sequence_layout"
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_rank", lambda: 0)
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_world_size", lambda: pcp_size)
    runner = _make_runner(
        num_computed_tokens=[4, 0],
        prompt_tokens=[8, 6],
        scheduled_tokens=[2, 6],
        token_paddings=[16],
        pcp_size=pcp_size,
        dp_size=dp_size,
        max_num_reqs=4,
    )
    runner._replay_starts = {"req0": 4}  # noqa: SLF001 -- seed bounded-replay start
    if dp_lockstep:
        runner.kv_cache_config = _swa_kv_config()
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([2, 6]), 0, 0, 0)  # noqa: SLF001 -- exercise _prepare_inputs hook
    assert runner._lookback_chunk_span == (0, 2)  # noqa: SLF001 -- verify chunk span recorded
    ctx = runner._attn_metadata_builder_ctx  # noqa: SLF001 -- inspect builder context
    expected_prep = (
        (
            (0, 2, 8, 8, 8),
            (6, 6, 0, 0),
            (8, 6, 0, 0),
            (4, 0, 0, 0) if dp_lockstep else (0, 0, 0, 0),
        )
        if pcp_size == 1
        else (None, None, None, None)
    )
    assert (
        ctx.query_start_loc_cpu,
        ctx.seq_lens_cpu,
        ctx.prompt_lens_cpu,
        ctx.replay_starts_cpu,
    ) == expected_prep

    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", lambda *_a, **_kw: None
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    runner._record_input_staging_fence = lambda: None  # noqa: SLF001 -- stub dummy fence
    runner._dp_lockstep_enabled = lambda: dp_lockstep  # noqa: SLF001 -- stub dp lockstep state
    runner._dp_num_tokens_across_dp = lambda _n: None  # noqa: SLF001 -- stub dp allreduce
    runner.compute_selected_logits = lambda out, _idx: out
    runner._profile_inputs = {}  # noqa: SLF001 -- initialize profile cache
    runner.vocab_size = 128
    runner._pp_topk_buffer = None  # noqa: SLF001 -- disable pp topk buffer
    runner._profile_input = MethodType(TPUModelRunner._profile_input, runner)  # noqa: SLF001 -- bind profile helper
    runner.forward_model = lambda *, positions, **_kw: (
        torch.zeros((positions.shape[-1], 4), dtype=torch.float32),
        None,
    )
    TPUModelRunner._dummy_run(  # noqa: SLF001 -- exercise _dummy_run hook
        runner,
        8,
        4,
        2,
        use_max_model_len=True,
        dp_lockstep=dp_lockstep,
        profile_prefix=0 if profiling else None,
    )
    assert runner._lookback_chunk_span == (0, 4)  # noqa: SLF001 -- verify dummy chunk span
    dummy_ctx = runner._attn_metadata_builder_ctx  # noqa: SLF001 -- inspect dummy builder context
    expected = (
        ((0, 0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        if dp_lockstep and not profiling
        else (None, None, None, None)
    )
    assert (
        dummy_ctx.query_start_loc_cpu,
        dummy_ctx.seq_lens_cpu,
        dummy_ctx.prompt_lens_cpu,
        dummy_ctx.replay_starts_cpu,
    ) == expected


def test_should_retain_swa_tokens_end_to_end_through_kv_configs_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "vllm_torchtpu:_patch_dflash_sliding_window_retained_tokens"
    reg_targets = {
        p.target for p in patch_registry.PATCHES if "engine_core" in p.stages
    }
    assert target in reg_targets

    applied = set(patch_registry._applied) - {target}  # noqa: SLF001 -- reset registry entry for clean apply test
    monkeypatch.setattr(patch_registry, "_applied", applied)
    orig_groups = inspect.unwrap(kv_cache_planning.get_kv_cache_groups)
    monkeypatch.setattr(kv_cache_planning, "get_kv_cache_groups", orig_groups)
    patch_registry.apply("engine_core")
    patched_groups = kv_cache_planning.get_kv_cache_groups
    assert patched_groups is not orig_groups
    # Second application must be idempotent rather than stacking wrappers.
    _patch_dflash_sliding_window_retained_tokens()
    assert kv_cache_planning.get_kv_cache_groups is patched_groups

    base_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    mla_swa = SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    high_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        extra_retained_tokens=10,
    )
    full_spec = FullAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    cache_cfg = CacheConfig(enable_prefix_caching=True)
    cache_cfg.kv_cache_layout = "LBHNC"
    base_vcfg = dict(
        cache_config=cache_cfg,
        model_config=SimpleNamespace(
            max_model_len=2048,
            original_max_model_len=2048,
            hf_config=SimpleNamespace(),
            supports_kv_cache_grouping=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_seqs=4
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=16,
    )

    groups_plain = kv_cache_planning.get_kv_cache_groups(
        SimpleNamespace(**base_vcfg), {"attn.0": base_swa}
    )
    assert groups_plain[0].kv_cache_spec.extra_retained_tokens == 0

    vcfg_zero_la = SimpleNamespace(
        **{
            **base_vcfg,
            "speculative_config": SimpleNamespace(
                method="dflash",
                use_eagle=lambda: False,
                use_eagle_block_drop=lambda: False,
            ),
            "num_lookahead_tokens": 0,
        }
    )
    groups_zero_la = kv_cache_planning.get_kv_cache_groups(
        vcfg_zero_la, {"attn.0": base_swa}
    )
    assert groups_zero_la[0].kv_cache_spec.extra_retained_tokens == 0

    vcfg_dspark = SimpleNamespace(
        **{
            **base_vcfg,
            "speculative_config": SimpleNamespace(
                method="dspark",
                use_eagle=lambda: False,
                use_eagle_block_drop=lambda: False,
            ),
            "num_lookahead_tokens": 5,
        }
    )
    groups_mixed = kv_cache_planning.get_kv_cache_groups(
        vcfg_dspark,
        {"swa.0": base_swa, "swa.1": high_swa, "full.0": full_spec},
    )
    by_layer = {g.layer_names[0]: g.kv_cache_spec for g in groups_mixed}
    assert by_layer["swa.0"].extra_retained_tokens == 6
    assert by_layer["swa.1"].extra_retained_tokens == 10

    # End-to-end Consumer(Producer(x)): upstream get_global_kv_cache_groups resets
    # extra_retained_tokens to 0 before get_kv_cache_groups, yet the patched
    # group retains 6 tokens and drives runner replay tracking + lookback forward.
    global_groups = kv_cache_planning.get_global_kv_cache_groups(
        vcfg_dspark, [{"attn.0": mla_swa}]
    )
    configs = kv_cache_planning.get_kv_cache_configs(
        vcfg_dspark, [{"attn.0": mla_swa}], [64 * 1024 * 1024], global_groups
    )
    kv_cfg = configs[0]
    retained_spec = kv_cfg.kv_cache_groups[0].kv_cache_spec
    assert retained_spec.extra_retained_tokens == 6

    mgr_stub = SimpleNamespace(
        sliding_window=retained_spec.sliding_window,
        extra_retained_tokens=retained_spec.extra_retained_tokens,
    )
    skipped = SlidingWindowManager.get_num_skipped_tokens(mgr_stub, 895)
    assert skipped == 762 and skipped // 32 == 23

    runner = _make_runner(
        num_computed_tokens=[6, 0],
        prompt_tokens=[4, 4],
        scheduled_tokens=[2, 4],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=4,
    )
    runner.kv_cache_config = kv_cfg
    runner._record_replay_starts = MethodType(  # noqa: SLF001 -- bind real replay tracker
        TPUModelRunner._record_replay_starts,  # noqa: SLF001 -- bind real replay tracker
        runner,
    )
    runner._prepare_lookback_token_ids = MethodType(  # noqa: SLF001 -- bind real lookback preparer
        TPUModelRunner._prepare_lookback_token_ids,  # noqa: SLF001 -- bind real lookback preparer
        runner,
    )
    runner._lookback_staging_ring = None  # noqa: SLF001 -- sync staging mode
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    buf_np = np.zeros((4, 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    runner.input_batch.num_tokens_no_spec = np.array([8, 4, 0, 0], dtype=np.int32)
    runner.input_batch.token_ids_cpu = np.array(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    runner._record_replay_starts(  # noqa: SLF001 -- record warm SWA replay hit from scheduler output
        SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="req0", replay_start=6)],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]
            ),
        )
    )
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([2, 4]), 0, 0, 0)  # noqa: SLF001 -- populate metadata context
    builder = AttentionMetadataBuilder(
        kv_cache_spec=retained_spec,
        layer_names=["attn.0"],
        vllm_config=MagicMock(),
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    built_meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert built_meta.query_start_loc_cpu == (0, 2, 6, 6, 6)
    assert built_meta.seq_lens_cpu == (8, 4, 0, 0)
    assert built_meta.prompt_lens_cpu == (4, 4, 0, 0)
    assert built_meta.replay_starts_cpu == (6, 0, 0, 0)

    model = _DummyLookbackModel(depth=3, uses_smem=False)
    runner.model = model
    TPUModelRunner.forward_model(
        runner,
        torch.zeros(6, dtype=torch.int32),
        torch.zeros(6, dtype=torch.int32),
    )
    lookback = model.last_kwargs["lookback_token_ids"]
    assert isinstance(lookback, torch.Tensor)
    assert lookback[0].tolist() == [15, 14, 13]
    assert lookback[1].tolist() == [-1, -1, -1]


@pytest.mark.tpu_test
def test_should_stage_fenced_lookback_ring_and_build_metadata_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only runtime extension imported inside tpu_test

    dev = torch.device("tpu:0")
    bt = SimpleNamespace(
        max_num_blocks_per_req=1,
        blocks_per_kv_block=1,
        get_cpu_tensor=lambda: torch.tensor([[3], [5], [7], [9]], dtype=torch.int32),
    )
    meta_runner = SimpleNamespace(
        device=dev,
        block_size=8,
        max_num_reqs=4,
        most_model_len=32,
        position_ids=torch.arange(4, dtype=torch.int32, device=dev),
        _block_table_stage_cpu=None,
        _free_h2d_sources={},
        _registered_h2d_sources=set(),
        _unfenced_h2d_sources=[],
        attn_groups=[],
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt]),
    )
    spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        page_size_padded=2048,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["state.0"],
        vllm_config=MagicMock(),
        device=dev,
        runner=meta_runner,
        kv_cache_group_id=0,
    )
    meta_runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(  # noqa: SLF001 -- test runner builder context
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=dev),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32, device=dev),
        query_start_loc_cpu=(0, 2, 4, 4, 4),
        seq_lens_cpu=(24, 16, 0, 0),
        prompt_lens_cpu=(20, 16, 0, 0),
        replay_starts_cpu=(8, 0, 0, 0),
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert meta.block_tables.device.type == "tpu"
    assert meta.block_tables.cpu().tolist() == [3, 5, 0, 0]
    assert meta.replay_starts_cpu == (8, 0, 0, 0)

    runner = _init_cpu_runner(monkeypatch)
    runner.device = dev
    runner.scheduler_config.async_scheduling = True
    ring_arrays = [
        np.zeros((4, 3), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)
    ]
    runner._lookback_staging_ring = tuple(  # noqa: SLF001 -- stub TPU ring buffers
        SimpleNamespace(
            np=arr,
            copy_to_gpu=lambda a=arr: torch.from_numpy(a).to(dev, non_blocking=True),
        )
        for arr in ring_arrays
    )
    runner._lookback_staging_fences = [None] * _LOOKBACK_RING_DEPTH  # noqa: SLF001 -- reset fences
    runner._lookback_staging_idx = 0  # noqa: SLF001 -- reset cursor
    runner._replay_starts = {"req1": 6}  # noqa: SLF001 -- seed warm replay start
    runner.input_batch = SimpleNamespace(
        req_ids=["req0", "req1"],
        num_computed_tokens_cpu=np.array([5, 6, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 4, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 7, 0, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    outputs: list[torch.Tensor] = []
    for step in range(6):
        runner.input_batch.token_ids_cpu[1, 5] = 25 + step
        outputs.append(runner._prepare_lookback_token_ids(2))  # noqa: SLF001 -- dispatch 6 fenced async steps
    assert all(f is not None for f in runner._lookback_staging_fences)  # noqa: SLF001 -- verify all 4 slots fenced
    for step, lb in enumerate(outputs):
        assert lb.device.type == "tpu"
        assert lb[0].cpu().tolist() == [-1, 13, 12]
        assert lb[1].cpu().tolist() == [25 + step, 24, 23]
