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
"""Unit and TPU tests for DeepSeek-V4.1 cache, CED replay, and MoE modules."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DecoderReplayPlan,
    DeepseekV41MoE,
    DsV41CacheMixin,
    PagedCacheRows,
    build_decoder_inverse_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    cache as cache_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    ced as ced_mod,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    moe as moe_mod,
)
from vllm_torchtpu.runner.kv_cache_dsv41 import DsV41CacheLayer, is_cache_for_ds_v41


@pytest.fixture
def mock_cache_ops(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _pack_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        return (
            torch.ones((n, 128), dtype=torch.int32),
            torch.ones((n, 32), dtype=torch.int32) * 2,
        )

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_gather_cache_rows", _gather)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _write)
    monkeypatch.setattr(cache_mod.cache_op, "dsv41_pack_fp8_kv_rows", _pack_fp8)


def test_should_reject_invalid_bank_or_layout_when_constructing_paged_cache_rows() -> (
    None
):
    raw = torch.zeros((8, 128), dtype=torch.int32)
    with pytest.raises(ValueError, match="Invalid native cache bank"):
        PagedCacheRows(
            torch.zeros((8, 64), dtype=torch.int32),
            ((2, 4, 128), torch.bfloat16, 1024, 0),
        )
    with pytest.raises(ValueError, match="Invalid native cache bank"):
        PagedCacheRows(raw, ((2, 4, 128), torch.int32, 2048, 0))
    with pytest.raises(ValueError, match="FP8 records"):
        PagedCacheRows(raw, ((2, 4, 512), torch.uint8, 2048, 0), rope=None)
    with pytest.raises(ValueError, match="FP8 records"):
        PagedCacheRows(
            torch.zeros((6, 128), dtype=torch.int32),
            ((1, 6, 512), torch.uint8, 3072, 0),
            rope=torch.zeros((1, 128), dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="FP8 records"):
        PagedCacheRows(
            raw,
            ((2, 4, 512), torch.uint8, 2048, 0),
            rope=torch.zeros((4, 128), dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="word/row aligned"):
        PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 1024, 128))
    with pytest.raises(ValueError, match="word/row aligned"):
        PagedCacheRows(raw, ((2, 4, 192), torch.bfloat16, 1536, 0))


def test_should_return_pages_only_when_block_aligned_in_paged_cache_rows() -> None:
    raw = torch.zeros((8, 128), dtype=torch.int32)
    pcr_unaligned = PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 2048, 512))
    assert pcr_unaligned.numel() == 1024 and pcr_unaligned.device.type == "cpu"
    assert pcr_unaligned.pages(torch.tensor([[0, 1]], dtype=torch.int32)) is None

    pcr_aligned = PagedCacheRows(raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    assert torch.equal(
        pcr_aligned.pages(torch.tensor([[0, 1]], dtype=torch.int64)),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )

    pcr_bad_r = PagedCacheRows(
        torch.zeros((6, 128), dtype=torch.int32), ((2, 3, 256), torch.bfloat16, 1536, 0)
    )
    assert pcr_bad_r.pages(torch.tensor([[0]], dtype=torch.int32)) is None

    pcr_bad_total = PagedCacheRows(
        torch.zeros((10, 128), dtype=torch.int32),
        ((2, 4, 256), torch.bfloat16, 2048, 0),
    )
    assert pcr_bad_total.pages(torch.tensor([[0, 1]], dtype=torch.int32)) is None


def test_should_update_persistent_bank_when_writing_sub_row_bf16(
    mock_cache_ops: None,
) -> None:
    del mock_cache_ops
    raw = torch.zeros((16, 128), dtype=torch.int32)
    layout = ((2, 4, 128), torch.bfloat16, 2048, 512)
    pcr_step1 = PagedCacheRows(raw, layout)
    vals = torch.randn(3, 128, dtype=torch.bfloat16)
    pcr_step1.write(torch.tensor([0, 4, -1], dtype=torch.int64), vals.float())
    assert pcr_step1.raw is raw
    pcr_step2 = PagedCacheRows(raw, layout)
    fetched = pcr_step2[torch.tensor([0, 1]), torch.tensor([0, 0])]
    assert torch.allclose(fetched[0], vals[0])
    assert torch.allclose(fetched[1], vals[1])


def test_should_roundtrip_multi_row_float32_when_reading_and_writing_paged_cache_rows(
    mock_cache_ops: None,
) -> None:
    del mock_cache_ops
    raw_f32 = torch.zeros((8, 128), dtype=torch.int32)
    pcr_f32 = PagedCacheRows(raw_f32, ((1, 4, 256), torch.float32, 4096, 0))
    vals_f32 = torch.randn(2, 256, dtype=torch.float32)
    pcr_f32.write(
        torch.tensor([0, 1], dtype=torch.int32),
        vals_f32,
        valid=torch.tensor([True, True]),
    )
    got_f32 = pcr_f32[torch.tensor([0, 0]), torch.tensor([0, 1])]
    assert torch.allclose(got_f32, vals_f32)


def test_should_update_raw_and_rope_banks_when_writing_fp8_paged_cache_rows(
    mock_cache_ops: None,
) -> None:
    del mock_cache_ops
    raw = torch.zeros((16, 128), dtype=torch.int32)
    rope = torch.zeros((4, 128), dtype=torch.int32)
    pcr_fp8 = PagedCacheRows(
        raw,
        ((4, 4, 512), torch.uint8, 2048, 0),
        rope=rope,
    )
    lat = torch.randn(2, 512, dtype=torch.bfloat16)
    pcr_fp8.write(
        torch.tensor([0, 1], dtype=torch.int32),
        lat,
        records=torch.tensor([0, 8], dtype=torch.int64),
    )
    assert pcr_fp8.raw is raw and pcr_fp8.rope is rope
    assert torch.all(raw[0] == 1) and torch.all(raw[8] == 1)
    assert torch.all(rope[0, :32] == 2) and torch.all(rope[2, :32] == 2)
    with pytest.raises(TypeError, match="FP8 row indexing"):
        _ = pcr_fp8[torch.tensor([0]), torch.tensor([0])]


def test_should_dispatch_paged_rows_and_dense_tensors_when_calling_write_cache(
    mock_cache_ops: None,
) -> None:
    del mock_cache_ops
    pcr = PagedCacheRows(
        torch.zeros((8, 128), dtype=torch.int32), ((1, 4, 256), torch.float32, 4096, 0)
    )
    vals = torch.randn(2, 256, dtype=torch.float32)
    valid = torch.tensor([True, False], dtype=torch.bool)
    cache_mod.write_cache(pcr, torch.tensor([0, 1]), vals, valid)
    cache_mod.write_cache(
        pcr, torch.tensor([0, 1]), vals, valid, records=torch.tensor([0, 1])
    )
    t_2d = torch.zeros((4, 256), dtype=torch.float32)
    cache_mod.write_cache(t_2d, torch.tensor([2, 3]), vals, valid)
    assert torch.allclose(t_2d[2], vals[0]) and torch.all(t_2d[3] == 0)
    t_3d = torch.zeros((2, 2, 256), dtype=torch.float32)
    cache_mod.write_cache(t_3d, torch.tensor([3, 0]), vals, valid)
    assert torch.allclose(t_3d[1, 1], vals[0])


def test_should_forward_and_reset_read_dependency_when_writing_after_read(
    mock_cache_ops: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del mock_cache_ops
    seen_after: list[torch.Tensor | None] = []
    seen_page_tokens: list[int] = []
    orig_write = cache_mod.cache_op.dsv41_write_sub_rows

    def _spy_write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        *,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        seen_after.append(None if after is None else after.clone())
        seen_page_tokens.append(page_tokens)
        return orig_write(bank, updates, slots, after=after, page_tokens=page_tokens)

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _spy_write)
    raw = torch.full((8, 128), 42, dtype=torch.int32)
    pcr = PagedCacheRows(raw, ((2, 4, 128), torch.bfloat16, 2048, 0))
    vals = torch.ones((2, 128), dtype=torch.bfloat16)
    valid = torch.tensor([True, True], dtype=torch.bool)
    slots = torch.tensor([0, 1], dtype=torch.int32)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    _ = pcr[torch.tensor([0]), torch.tensor([0])]
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is None

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _ = pcr[torch.tensor([1]), torch.tensor([0])]
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is not None and int(seen_after[-1].item()) == 42
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is None

    _ = pcr[torch.tensor([1]), torch.tensor([1])]
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=False)
    assert seen_after[-1] is None
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after[-1] is None
    assert seen_page_tokens == [4, 4, 4, 4, 4]


def test_should_emit_ordered_gather_and_write_after_when_compiled_fullgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = torch.zeros((8, 128), dtype=torch.int32)
    pcr = PagedCacheRows(raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    vals = torch.ones((2, 256), dtype=torch.bfloat16)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    valid = torch.tensor([True, True], dtype=torch.bool)
    captured_ops: list[str] = []

    def _capture(gm: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Any:
        captured_ops.extend(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        return lambda *_a: (torch.zeros((1, 256), dtype=torch.bfloat16),)

    def _read_then_write(
        vals_in: torch.Tensor, s_in: torch.Tensor, v_in: torch.Tensor
    ) -> torch.Tensor:
        pre = pcr[s_in[:1], s_in[:1]]
        cache_mod.write_cache(pcr, s_in, vals_in, v_in, after_read=True)
        return pre

    torch.compiler.reset()
    compiled_fn = torch.compile(
        _read_then_write, backend=_capture, fullgraph=True, dynamic=False
    )
    out = compiled_fn(vals, slots, valid)
    assert out.shape == (1, 256)
    assert "pallas.dsv41_gather_cache_rows.default" in captured_ops
    assert "pallas.dsv41_write_sub_rows_after.default" in captured_ops
    assert captured_ops.index(
        "pallas.dsv41_gather_cache_rows.default"
    ) < captured_ops.index("pallas.dsv41_write_sub_rows_after.default")

    seen_after: list[torch.Tensor | None] = []
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        cache_mod.cache_op,
        "dsv41_write_sub_rows",
        lambda bank, _u, _s, *, after=None, page_tokens=16: (
            seen_after.append(after) or bank
        ),
    )
    cache_mod.write_cache(pcr, slots, vals, valid, after_read=True)
    assert seen_after == [None]


def test_should_bind_and_zero_3d_single_bank_pools_when_using_dsv41_cache_mixin(
    mock_cache_ops: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del mock_cache_ops
    layer = DsV41CacheMixin()
    assert isinstance(layer, DsV41CacheLayer)
    assert is_cache_for_ds_v41(layer) is True
    assert layer.paged_cache_rows is None and layer.kv_cache is None
    assert layer.null_block is False

    layer.zero_blocks([1])
    layer.kv_cache = (torch.ones((2, 4)),)
    layer.zero_blocks([1])

    dense = torch.ones((3, 4, 256), dtype=torch.bfloat16)
    raw_3d = torch.zeros((3, 4, 128), dtype=torch.int32)
    layer.kv_cache = dense
    assert layer.kv_cache is dense
    layer.zero_blocks([0, 99])
    assert float(dense[1, 0, 0].item()) == 1.0
    layer.zero_blocks([1])
    assert float(dense[0, 0, 0].item()) == 1.0 and float(dense[1, 0, 0].item()) == 0.0

    layer.bind_kv_cache(dense, raw_cache=None)
    assert layer.paged_cache_rows is None
    layer.bind_kv_cache(dense, raw_cache=raw_3d)
    assert isinstance(layer.paged_cache_rows, PagedCacheRows)
    assert layer.null_block is True
    layer.kv_cache = dense
    assert layer.paged_cache_rows is not None

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert isinstance(layer.kv_cache, PagedCacheRows)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert layer.kv_cache is dense

    raw_half = torch.ones((2, 4, 128), dtype=torch.int32)
    layer.bind_kv_cache(
        dense,
        raw_cache=raw_half,
        layout=((4, 4, 256), torch.bfloat16, 2048, 0),
    )
    layer.zero_blocks([0])
    assert int(raw_half[1, 0, 0].item()) == 1
    layer.zero_blocks([1, 2])
    assert int(raw_half[0].min().item()) == 1
    assert int(raw_half[1].abs().max().item()) == 0

    fp8_raw = torch.full((4, 8, 128), 7, dtype=torch.int32)
    fp8_rope = torch.full((4, 2, 128), 11, dtype=torch.int32)
    fp8_layer = DsV41CacheMixin()
    fp8_layer.bind_kv_cache(
        (fp8_raw, fp8_rope),
        raw_cache=fp8_raw,
        layout=((4, 8, 512), torch.uint8, 4096, 0),
        rope_pool=fp8_rope,
    )
    fp8_layer.zero_blocks([2])
    assert int(fp8_raw[0, 0, 0].item()) == 7 and int(fp8_raw[2].abs().max().item()) == 0
    assert int(fp8_rope[0, 0, 0].item()) == 11
    assert int(fp8_rope[2].abs().max().item()) == 0


def test_should_validate_prefix_replay_flags_and_decoder_replay_metadata() -> None:
    def _cfg(
        *,
        prefix_caching: bool = True,
        bounded_replay: bool = True,
        pcp_size: int = 1,
    ) -> Any:
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_prefix_caching=prefix_caching,
                swa_bounded_replay=bounded_replay,
            ),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        )

    assert ced_mod.prefix_replay_enabled(_cfg()) is True
    assert ced_mod.prefix_replay_enabled(_cfg(), ced_enabled=False) is False
    assert ced_mod.prefix_replay_enabled(_cfg(prefix_caching=False)) is False
    assert ced_mod.prefix_replay_enabled(_cfg(bounded_replay=False)) is False
    assert ced_mod.prefix_replay_enabled(_cfg(pcp_size=2)) is False

    assert (
        ced_mod.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=None, seq_lens_cpu=None, prompt_lens_cpu=None
            ),
            4,
            8,
            (4, 8),
        )
        is None
    )
    bad_meta = SimpleNamespace(
        query_start_loc_cpu=[0, 10],
        seq_lens_cpu=[4],
        prompt_lens_cpu=[4],
        replay_starts_cpu=None,
    )
    with pytest.raises(ValueError, match="Invalid decoder replay scheduling"):
        ced_mod.plan_decoder_replay(bad_meta, 4, 8, (4, 8))
    bad_q = SimpleNamespace(
        query_start_loc_cpu=[0, 5],
        seq_lens_cpu=[3],
        prompt_lens_cpu=[3],
        replay_starts_cpu=[0],
    )
    with pytest.raises(ValueError, match="Invalid scheduled query"):
        ced_mod.plan_decoder_replay(bad_q, 4, 8, (4, 8))


def test_should_sync_dp_replay_plans_and_scatter_any_rank_without_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 6, 7], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([6, 10], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([6, 9], dtype=torch.int32),
        replay_starts_cpu=None,
        seq_lens=torch.tensor([6, 10], dtype=torch.int32),
    )
    plan = ced_mod.plan_decoder_replay(
        meta, window=4, num_tokens=8, compile_sizes=[4, 8]
    )
    assert plan is not None and plan.indices == (2, 3, 4, 5, 6)
    assert plan.padded_num_tokens == 8 and not plan.is_identity

    meta_hit = SimpleNamespace(
        query_start_loc_cpu=[0, 6],
        seq_lens_cpu=[6],
        prompt_lens_cpu=[6],
        replay_starts_cpu=[4],
        seq_lens=torch.tensor([6], dtype=torch.int32),
    )
    plan_hit = ced_mod.plan_decoder_replay(
        meta_hit, window=4, num_tokens=8, compile_sizes=[2, 4, 8]
    )
    assert plan_hit is not None
    assert plan_hit.replay_starts == (4,)
    assert plan_hit.indices == (4, 5)
    assert plan_hit.padded_num_tokens == 2

    meta_skip = SimpleNamespace(
        query_start_loc_cpu=[0, 4],
        seq_lens_cpu=[4],
        prompt_lens_cpu=[20],
        replay_starts_cpu=[0],
    )
    plan_skip = ced_mod.plan_decoder_replay(
        meta_skip, window=4, num_tokens=8, compile_sizes=[4, 8]
    )
    assert plan_skip is not None
    assert plan_skip.indices == () and plan_skip.padded_num_tokens == 0

    meta_id = SimpleNamespace(
        query_start_loc_cpu=[0, 4],
        seq_lens_cpu=[4],
        prompt_lens_cpu=[4],
        replay_starts_cpu=[0],
    )
    plan_id = ced_mod.plan_decoder_replay(meta_id, 4, 4, [4])
    assert plan_id is not None and plan_id.is_identity

    assert ced_mod.sync_decoder_replay_plan(plan, meta, 8, None) is plan

    reduce_calls: list[Any] = []
    peer_state = [1, 8]

    def _peer_all_reduce(buf: torch.Tensor, op: Any = None, group: Any = None) -> None:
        del group
        reduce_calls.append(op)
        peer = torch.tensor(peer_state, dtype=buf.dtype)
        if op == ced_mod.dist.ReduceOp.MAX:
            torch.maximum(buf, peer, out=buf)
        elif op == ced_mod.dist.ReduceOp.MIN:
            torch.minimum(buf, peer, out=buf)
        else:
            raise AssertionError(f"Unexpected ReduceOp: {op}")

    monkeypatch.setattr(ced_mod.dist, "all_reduce", _peer_all_reduce)
    s_fast_active = ced_mod.sync_decoder_replay_plan(
        plan, meta, 4, object(), window=4, prefix_replay=False, compile_sizes=(4, 8)
    )
    s_fast_idle = ced_mod.sync_decoder_replay_plan(
        None, meta, 4, object(), window=4, prefix_replay=False, compile_sizes=(4, 8)
    )
    assert reduce_calls == []
    assert s_fast_active is not None and s_fast_active.padded_num_tokens == 4
    assert s_fast_idle is not None and s_fast_idle.padded_num_tokens == 4

    peer_state = [1, 8]
    s_upgraded = ced_mod.sync_decoder_replay_plan(
        plan_hit, meta_hit, 8, object(), prefix_replay=True
    )
    assert s_upgraded is not None and s_upgraded.padded_num_tokens == 8
    s_none = ced_mod.sync_decoder_replay_plan(
        None, meta, 8, object(), prefix_replay=True
    )
    assert isinstance(s_none, DecoderReplayPlan) and s_none.padded_num_tokens == 8

    peer_state = [1, 4]
    assert (
        ced_mod.sync_decoder_replay_plan(plan, meta, 8, object(), prefix_replay=True)
        is plan
    )

    peer_state = [0, 0]
    assert (
        ced_mod.sync_decoder_replay_plan(None, meta, 8, object(), prefix_replay=True)
        is None
    )

    inv = build_decoder_inverse_rows((1, 3), padded=4, total=5, device="cpu")
    assert inv.tolist() == [4, 0, 4, 1, 4]

    narrow_1d = torch.arange(4, dtype=torch.float32) + 1.0
    scattered_1d = ced_mod.scatter_decoder_outputs(narrow_1d, inv)
    assert torch.equal(scattered_1d, torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0]))

    narrow_2d = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2) + 1.0
    scattered_2d = ced_mod.scatter_decoder_outputs(narrow_2d, inv)
    assert scattered_2d.shape == (5, 2)
    assert torch.equal(scattered_2d[0], torch.zeros(2))
    assert torch.equal(scattered_2d[1], narrow_2d[0])
    assert torch.equal(scattered_2d[3], narrow_2d[1])

    narrow_3d = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2) + 1.0
    scattered_3d = ced_mod.scatter_decoder_outputs(narrow_3d, inv)
    assert scattered_3d.shape == (5, 3, 2)
    assert torch.equal(scattered_3d[0], torch.zeros((3, 2)))
    assert torch.equal(scattered_3d[1], narrow_3d[0])
    assert torch.equal(scattered_3d[2], torch.zeros((3, 2)))
    assert torch.equal(scattered_3d[3], narrow_3d[1])
    assert torch.equal(scattered_3d[4], torch.zeros((3, 2)))

    inv_3d_id = build_decoder_inverse_rows((0, 1), padded=2, total=2, device="cpu")
    narrow_3d_id = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) + 1.0
    scattered_3d_id = ced_mod.scatter_decoder_outputs(narrow_3d_id, inv_3d_id)
    assert scattered_3d_id.shape == (2, 3, 4)
    assert torch.equal(scattered_3d_id, narrow_3d_id)

    with pytest.raises(ValueError, match="at least a 1D tensor"):
        ced_mod.scatter_decoder_outputs(torch.tensor(1.0), inv)


def test_should_build_self_contained_v41_moe_and_slice_ep_gate_inputs(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")

    def _make_stub_cfg(eplb: bool = False, backend: str = "auto") -> Any:
        return SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=eplb),
            kernel_config=SimpleNamespace(moe_backend=backend),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )

    with pytest.raises(NotImplementedError, match="EPLB"):
        DeepseekV41MoE(_make_stub_cfg(eplb=True))
    with pytest.raises(NotImplementedError, match="native V4 MoE"):
        DeepseekV41MoE(_make_stub_cfg(backend="deep_gemm_mega_moe"))

    def _make_real_vcfg(text_overrides: dict[str, Any]) -> VllmConfig:
        base = {
            "hidden_size": 16,
            "num_hidden_layers": 40,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "n_shared_experts": None,
            "swiglu_limit": None,
            "norm_topk_prob": True,
            "hidden_act": "silu",
            "vocab_size": 32,
        }
        base.update(text_overrides)
        hf_cfg = DeepseekV41Config(text_config=base)
        vcfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        vcfg.model_config = SimpleNamespace(
            hf_config=hf_cfg,
            dtype=torch.bfloat16,
            max_model_len=64,
        )
        vcfg.quant_config = None
        return vcfg

    # Non-silu shared MLP activation fails fast.
    bad_act_vcfg = _make_real_vcfg({"n_shared_experts": 1, "hidden_act": "gelu"})
    with (
        set_current_vllm_config(bad_act_vcfg),
        pytest.raises(ValueError, match="Unsupported activation"),
    ):
        DeepseekV41MoE(bad_act_vcfg, prefix="layers.0.ffn")

    # Routed experts not divisible by TP size fails fast.
    bad_tp_vcfg = _make_real_vcfg({"n_routed_experts": 3})
    with (
        monkeypatch.context() as mp,
        set_current_vllm_config(bad_tp_vcfg),
        pytest.raises(ValueError, match="divisible by tp_size"),
    ):
        mp.setattr(moe_mod, "get_tensor_model_parallel_world_size", lambda: 2)
        DeepseekV41MoE(bad_tp_vcfg, prefix="layers.0.ffn")

    # Shared experts with clamp + noaux_tc correction bias + draft layer override.
    shared_clamped_vcfg = _make_real_vcfg(
        {
            "n_shared_experts": 1,
            "swiglu_limit": 7.0,
            "topk_method": "noaux_tc",
            "dspark_n_routed_experts": 4,
            "dspark_num_experts_per_tok": 2,
        }
    )
    with set_current_vllm_config(shared_clamped_vcfg):
        m_draft = DeepseekV41MoE(
            shared_clamped_vcfg, prefix="layers.40.ffn", reduce_results=False
        )
    assert m_draft.use_native_mega_moe is False
    assert m_draft.n_redundant_experts == 0
    assert m_draft.reduce_results is False
    assert m_draft.experts.moe_config.skip_final_all_reduce is True
    assert m_draft.n_routed_experts == 4
    assert m_draft.gate.e_score_correction_bias is not None
    assert m_draft.shared_experts is not None
    shared_out = m_draft.shared_experts(torch.ones((2, 16), dtype=torch.float32))
    assert shared_out.shape == (2, 16)

    # Shared experts without clamp (SiluAndMul) on backbone layer.
    shared_plain_vcfg = _make_real_vcfg({"n_shared_experts": 1, "swiglu_limit": None})
    with set_current_vllm_config(shared_plain_vcfg):
        m_shared = DeepseekV41MoE(shared_plain_vcfg, prefix="layers.2.ffn")
    assert m_shared.shared_experts is not None
    assert m_shared.gate.e_score_correction_bias is None
    assert m_shared.shared_experts(torch.ones((2, 16), dtype=torch.float32)).shape == (
        2,
        16,
    )

    # No shared experts + EP / DP gate slicing.
    real_vcfg = _make_real_vcfg({})
    with set_current_vllm_config(real_vcfg):
        m = DeepseekV41MoE(real_vcfg, prefix="layers.0.ffn")

    assert m.use_native_mega_moe is False
    assert m.n_redundant_experts == 0
    assert m.shared_experts is None
    assert m.experts.gate is None

    seen_gate_inputs: list[torch.Tensor] = []

    class _SpyGate(torch.nn.Module):
        def forward(self, gate_in: torch.Tensor) -> tuple[torch.Tensor, None]:
            seen_gate_inputs.append(gate_in.clone())
            return gate_in[:, :2], None

    class _StubExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe_config = SimpleNamespace(
                dp_size=1, pcp_size=1, moe_parallel_config=SimpleNamespace(use_ep=True)
            )
            self.routed_experts = SimpleNamespace(
                quant_method=SimpleNamespace(
                    is_monolithic=True, supports_internal_mk=False
                ),
                custom_routing_function=None,
            )

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del router_logits, input_ids
            return torch.ones_like(hidden_states)

    fake_experts = _StubExperts()
    m.gate = _SpyGate()
    m.experts = fake_experts

    monkeypatch.setattr(
        moe_mod, "get_ep_group", lambda: SimpleNamespace(world_size=2, rank_in_group=1)
    )
    x = torch.arange(64, dtype=torch.bfloat16).reshape(4, 16)
    out = m(x)
    assert out.shape == (4, 16)
    assert torch.equal(seen_gate_inputs[-1], x[2:4])

    with pytest.raises(ValueError, match="divisible by group size"):
        m(x[:3])

    fake_experts.moe_config.dp_size = 2
    monkeypatch.setattr(
        moe_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    out_dp = m(x)
    assert out_dp.shape == (4, 16)
    assert torch.equal(seen_gate_inputs[-1], x)


def test_should_isolate_null_block_nan_patterns_when_bound(
    mock_cache_ops: None,
) -> None:
    del mock_cache_ops

    class _DummyNativeCache(cache_mod.DsV41CacheMixin):
        pass

    nan_word = int(torch.tensor([0x7FC07FC0], dtype=torch.uint32).view(torch.int32)[0])
    raw = torch.full((32, 128), nan_word, dtype=torch.int32)
    dummy_view = torch.zeros((4, 4, 256), dtype=torch.bfloat16)
    layer1 = _DummyNativeCache()
    layer1.bind_kv_cache(
        dummy_view,
        raw_cache=raw,
        layout=((4, 4, 256), torch.bfloat16, 4096, 2048),
    )
    pcr = layer1.paged_cache_rows
    assert pcr is not None and pcr.null_block is True

    bt = torch.tensor([[-1, 0, 1, 3]], dtype=torch.int32)
    mapped_pages = pcr.pages(bt)
    assert mapped_pages is not None
    assert mapped_pages.tolist() == [[-1, -1, 3, 7]]

    blk0_before = raw[:8].clone()
    vals = torch.full((4, 256), 5.0, dtype=torch.bfloat16)
    cache_mod.write_cache(
        pcr,
        torch.tensor([-1, 0, 3, 13], dtype=torch.int32),
        vals,
        torch.ones(4, dtype=torch.bool),
    )
    assert torch.equal(raw[:8], blk0_before)

    gathered = pcr[
        torch.tensor([-1, 0, 3], dtype=torch.int32),
        torch.tensor([0, 2, 1], dtype=torch.int32),
    ]
    assert not torch.isnan(gathered).any()
    assert bool((gathered[0] == 0).all() and (gathered[1] == 0).all())
    assert bool((gathered[2] == 5.0).all())


@pytest.mark.tpu_test
def test_tpu_should_donate_paged_cache_writes_and_scatter_on_tpu() -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only test

    dev = torch.device("tpu:0")
    raw = torch.zeros((8, 128), dtype=torch.int32, device=dev)
    rope = torch.zeros((2, 128), dtype=torch.int32, device=dev)
    pcr = PagedCacheRows(raw, ((2, 4, 512), torch.uint8, 2048, 0), rope=rope)
    lat = torch.randn(4, 512, dtype=torch.bfloat16, device=dev)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
    pcr.write(slots, lat)
    pcr.write(torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=dev), lat)
    raw_cpu = pcr.raw.cpu()
    assert raw_cpu[:4].abs().sum().item() > 0
    assert raw_cpu[4:8].abs().sum().item() > 0

    bf16_raw = torch.zeros((8, 128), dtype=torch.int32, device=dev)
    bf16_pcr = PagedCacheRows(bf16_raw, ((2, 4, 256), torch.bfloat16, 2048, 0))
    bf16_vals = torch.randn(4, 256, dtype=torch.bfloat16, device=dev)
    bf16_pcr.write(slots, bf16_vals)
    bf16_got = bf16_pcr[
        torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=dev),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev),
    ]
    assert torch.allclose(bf16_got.cpu(), bf16_vals.cpu())

    inv = build_decoder_inverse_rows((1, 3), padded=4, total=5, device=dev)
    narrow = (
        torch.arange(4 * 2, dtype=torch.float32, device=dev).reshape(4, 2).bfloat16()
    )
    out = ced_mod.scatter_decoder_outputs(narrow, inv).cpu()
    assert out.shape == (5, 2)
    assert torch.allclose(out[1], narrow[0].cpu())
    assert torch.allclose(out[3], narrow[1].cpu())
    assert torch.all(out[0] == 0)

    def _compile_step(
        v_in: torch.Tensor, s_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = bf16_pcr[s_in[:1] // 4, s_in[:1] % 4]
        cache_mod.write_cache(
            bf16_pcr,
            s_in,
            v_in,
            torch.ones(4, dtype=torch.bool, device=dev),
            after_read=True,
        )
        return gathered, ced_mod.scatter_decoder_outputs(narrow, inv)

    compiled_step = torch.compile(_compile_step, backend="tpu", fullgraph=True)
    g_out, s_out = compiled_step(bf16_vals * 2.0, slots)
    assert torch.allclose(g_out.cpu(), bf16_vals[:1].cpu())
    assert torch.allclose(s_out.cpu(), out)
