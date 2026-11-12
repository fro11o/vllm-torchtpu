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
"""CPU reference and TPU7x parity tests for PagedCacheRows SWA window gathers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import cache as cache_mod
from vllm_torchtpu.models.vllm.deepseek_v41.cache import PagedCacheRows

HEAD_DIM = 512
PAGE_TOKENS = 16
RECORD_BYTES = HEAD_DIM * 2
WINDOW = 128


@pytest.fixture
def stub_window_op(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[torch.Tensor, tuple[int, ...]]]]:
    seen_calls: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def _bridge(
        pool: torch.Tensor,
        page_ids: torch.Tensor,
        phase_rows: torch.Tensor,
        *,
        window_rows: int,
        rows_per_page: int,
        page_row_stride: int,
        page_row_offset: int = 0,
    ) -> torch.Tensor:
        seen_calls.append((pool, tuple(page_ids.shape)))
        pool_2d = pool.reshape(-1, 128)
        num_seqs, pages_per_window = page_ids.shape
        out = torch.zeros(
            (num_seqs, window_rows, 128), dtype=torch.int32, device=pool.device
        )
        for seq in range(num_seqs):
            phase = int(phase_rows[seq].item())
            assert phase + window_rows <= pages_per_window * rows_per_page, (
                f"pages_per_window={pages_per_window} under-sized for "
                f"phase={phase}, window_rows={window_rows}, "
                f"rows_per_page={rows_per_page}"
            )
            for row_idx in range(window_rows):
                rel = phase + row_idx
                p_col = rel // rows_per_page
                in_page = rel % rows_per_page
                pid = int(page_ids[seq, p_col].item())
                if pid >= 0:
                    out[seq, row_idx] = pool_2d[
                        pid * page_row_stride + page_row_offset + in_page
                    ]
        return out

    monkeypatch.setattr(cache_mod.swa_window_op, "dsv41_gather_swa_windows", _bridge)
    yield seen_calls


def _bf16_bits_from_words(words: np.ndarray) -> np.ndarray:
    lo = (words & 0xFFFF).astype(np.uint16)
    hi = ((words >> 16) & 0xFFFF).astype(np.uint16)
    return np.stack((lo, hi), axis=-1).reshape(*words.shape[:-1], -1)


def _make_cache(
    num_blocks: int,
    page_stride_records: int,
    row_offset_records: int,
    *,
    page_tokens: int = PAGE_TOKENS,
    as_3d: bool = False,
    seed: int = 0,
) -> tuple[PagedCacheRows, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows_per_record = RECORD_BYTES // 512
    if as_3d:
        assert row_offset_records == 0
        raw = rng.integers(
            -(1 << 30),
            1 << 30,
            (num_blocks, page_stride_records * rows_per_record, 128),
        )
    else:
        total_records = num_blocks * page_stride_records + row_offset_records
        raw = rng.integers(-(1 << 30), 1 << 30, (total_records * rows_per_record, 128))
    # Avoid NaN and denormal bf16 bit patterns: f32 round-trips canonicalize NaN
    # payloads and TPU flushes denormals to zero (exponent forced into [1, 127]).
    raw &= ~((1 << 14) | (1 << 30))
    raw |= (1 << 7) | (1 << 23)
    raw_t = torch.from_numpy(raw.astype(np.int32))
    layout = (
        (num_blocks, page_tokens, HEAD_DIM),
        torch.bfloat16,
        page_stride_records * RECORD_BYTES,
        row_offset_records * RECORD_BYTES,
    )
    return PagedCacheRows(raw_t, layout, null_block=True), raw


def _expected_bits(
    raw: np.ndarray,
    table: np.ndarray,
    window_base: np.ndarray,
    page_stride_records: int,
    row_offset_records: int,
    *,
    page_tokens: int = PAGE_TOKENS,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    num_seqs = table.shape[0]
    words = raw.reshape(-1, HEAD_DIM // 2)
    out = np.zeros((num_seqs, window, HEAD_DIM), np.uint16)
    valid = np.zeros((num_seqs, window), bool)
    for seq in range(num_seqs):
        for offset in range(window):
            pos = window_base[seq] + offset
            col = pos // page_tokens
            if pos < 0 or col >= table.shape[1] or table[seq, col] < 1:
                continue
            record = (
                table[seq, col] * page_stride_records
                + row_offset_records
                + pos % page_tokens
            )
            out[seq, offset] = _bf16_bits_from_words(words[record])
            valid[seq, offset] = True
    return out, valid


@pytest.mark.parametrize(
    ("page_stride_records", "row_offset_records", "as_3d"),
    [
        (PAGE_TOKENS, 0, False),
        (PAGE_TOKENS, 0, True),
        (3 * PAGE_TOKENS, PAGE_TOKENS + 4, False),
    ],
)
def test_should_match_record_oracle_when_gathering_paged_windows(
    stub_window_op: list[tuple[torch.Tensor, tuple[int, ...]]],
    page_stride_records: int,
    row_offset_records: int,
    as_3d: bool,
) -> None:
    num_blocks = 64
    cache, raw = _make_cache(
        num_blocks, page_stride_records, row_offset_records, as_3d=as_3d
    )
    rng = np.random.default_rng(1)
    table = rng.permutation(np.arange(1, num_blocks))[: 5 * 12].reshape(5, 12)
    table[1, 3] = 0
    window_base = np.array([-37, 5, 61, 79, 64])

    got = cache.gather_paged_windows(
        torch.from_numpy(table.astype(np.int32)), torch.from_numpy(window_base), WINDOW
    )

    assert stub_window_op[-1][0] is cache.raw
    expected, valid = _expected_bits(
        raw, table, window_base, page_stride_records, row_offset_records
    )
    got_bits = got.view(torch.int16).numpy().view(np.uint16)
    assert got.shape == (5, WINDOW, HEAD_DIM)
    assert (~valid).any() and valid.any()
    np.testing.assert_array_equal(got_bits[valid], expected[valid])


def test_should_size_row_granular_pages_per_window_when_one_record_per_page(
    stub_window_op: list[tuple[torch.Tensor, tuple[int, ...]]],
) -> None:
    # records_per_page=1, window_records=4, rows_per_record=2 -> window_rows=8,
    # rows_per_page=2 -> pages_per_window = ceil((8 - 1) / 2) + 1 = 5 (covers odd
    # row phases; naive record-granular ceil((4 - 1) / 1) + 1 == 4 would under-size).
    cache, raw = _make_cache(12, 1, 0, page_tokens=1, as_3d=True, seed=3)
    table = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 0, 9, 10, 11]], dtype=np.int32)
    window_base = np.array([1, 0], dtype=np.int32)

    got = cache.gather_paged_windows(
        torch.from_numpy(table), torch.from_numpy(window_base), window_records=4
    )

    assert stub_window_op[-1][1] == (2, 5)
    expected, valid = _expected_bits(
        raw, table, window_base, 1, 0, page_tokens=1, window=4
    )
    got_bits = got.view(torch.int16).numpy().view(np.uint16)
    assert got.shape == (2, 4, HEAD_DIM)
    assert valid.any() and (~valid).any()
    np.testing.assert_array_equal(got_bits[valid], expected[valid])


def test_should_record_last_read_barrier_when_gathering_paged_windows_under_compile(
    stub_window_op: list[tuple[torch.Tensor, tuple[int, ...]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stub_window_op
    seen_after: list[torch.Tensor | None] = []

    def _spy_write(
        bank: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        *,
        after: torch.Tensor | None = None,
        page_tokens: int = 16,
    ) -> torch.Tensor:
        del updates, slots, page_tokens
        seen_after.append(None if after is None else after.clone())
        return bank

    monkeypatch.setattr(cache_mod.cache_op, "dsv41_write_sub_rows", _spy_write)
    cache, _ = _make_cache(16, PAGE_TOKENS, 0, as_3d=True, seed=7)
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    window_base = torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _ = cache.gather_paged_windows(table, window_base, 32)
    cache.write(
        torch.tensor([16], dtype=torch.int32),
        torch.ones((1, HEAD_DIM), dtype=torch.bfloat16),
        after_read=True,
    )
    assert seen_after[-1] is not None and seen_after[-1].shape == (1,)


def test_should_report_unsupported_when_pool_is_not_row_aligned_bf16() -> None:
    raw = torch.zeros((64, 128), dtype=torch.int32)
    rope = torch.zeros((16, 128), dtype=torch.int32)
    fp8 = PagedCacheRows(raw, ((4, PAGE_TOKENS, 512), torch.uint8, 8192, 0), rope=rope)
    sub_row = PagedCacheRows(raw, ((4, PAGE_TOKENS, 128), torch.bfloat16, 4096, 0))
    f32 = PagedCacheRows(raw, ((4, PAGE_TOKENS, 128), torch.float32, 8192, 0))
    contiguous = PagedCacheRows(
        raw, ((2, PAGE_TOKENS, HEAD_DIM), torch.bfloat16, 16384, 0)
    )

    assert not fp8.supports_paged_windows
    assert not sub_row.supports_paged_windows
    assert not f32.supports_paged_windows
    assert contiguous.supports_paged_windows


@pytest.mark.tpu_test
@pytest.mark.parametrize("as_3d", [False, True])
def test_tpu_should_match_getitem_when_paged_windows(as_3d: bool) -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only test

    dev = torch.device("tpu:0")
    num_blocks = 64
    cpu_cache, raw = _make_cache(num_blocks, PAGE_TOKENS, 0, as_3d=as_3d)
    cache = PagedCacheRows(
        cpu_cache.raw.to(dev),
        ((num_blocks, PAGE_TOKENS, HEAD_DIM), torch.bfloat16, 16 * RECORD_BYTES, 0),
        null_block=True,
    )
    rng = np.random.default_rng(2)
    table = rng.permutation(np.arange(1, num_blocks))[:48].reshape(4, 12)
    window_base = np.array([-37, 5, 61, 79])
    table_t = torch.from_numpy(table.astype(np.int64)).to(dev)
    base_t = torch.from_numpy(window_base).to(dev)
    pos = base_t[:, None] + torch.arange(WINDOW, device=dev)
    blocks = table_t.gather(1, (pos.clamp_min(0) // PAGE_TOKENS).clamp_max(11))

    paged = cache.gather_paged_windows(table_t, base_t, WINDOW).cpu()
    flat = cache[blocks, pos.clamp_min(0) % PAGE_TOKENS].cpu()

    expected, valid = _expected_bits(raw, table, window_base, PAGE_TOKENS, 0)
    for got in (paged, flat):
        got_bits = got.view(torch.int16).numpy().view(np.uint16)
        np.testing.assert_array_equal(got_bits[valid], expected[valid])
