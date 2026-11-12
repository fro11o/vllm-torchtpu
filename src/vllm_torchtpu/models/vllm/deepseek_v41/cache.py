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
"""DeepSeek-V4.1 single-bank paged KV row addressing and in-place writes."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    cache_op,
    swa_window_op,
)
from vllm_torchtpu.runner.kv_cache_dsv41 import (
    ROPE_COMPANION_DIVISOR,
    ROPE_WORDS,
    ROW_BYTES,
    ROW_WORDS,
    WORD_BYTES,
    DsV41CacheLayer,
    RawCachePool,
)

PAGE_ROW_ALIGNMENT: int = 4
BF16_SHIFT_BITS: int = 16
BF16_EVEN_MASK: int = 0xFFFF
BF16_ODD_MASK: int = -65536


def _bank_word_rows(bank: Tensor) -> int:
    return bank.numel() // ROW_WORDS


def _is_valid_word_bank(bank: Tensor) -> bool:
    return (
        bank.dtype == torch.int32
        and bank.ndim in (2, 3)
        and bank.shape[-1] == ROW_WORDS
    )


def _expand_chunks(records: Tensor, chunks_per_record: int) -> Tensor:
    offsets = torch.arange(chunks_per_record, dtype=torch.int32, device=records.device)
    expanded = torch.where(
        records[:, None] >= 0,
        records[:, None] * chunks_per_record + offsets,
        -1,
    )
    return expanded.reshape(-1)


def _unpack_bf16_record_words(words: Tensor) -> Tensor:
    even_lanes = (words << BF16_SHIFT_BITS).view(torch.float32).to(torch.bfloat16)
    odd_lanes = (words & BF16_ODD_MASK).view(torch.float32).to(torch.bfloat16)
    return torch.stack((even_lanes, odd_lanes), dim=-1).flatten(-2)


class PagedCacheRows:
    """Map logical page slots onto single-bank ``(R, 128)`` or ``(N, P, 128)`` pools."""

    def __init__(
        self,
        raw: RawCachePool,
        layout: tuple[tuple[int, ...], torch.dtype, int, int],
        rope: RawCachePool | None = None,
        *,
        null_block: bool = False,
    ) -> None:
        self._last_read: Tensor | None = None
        self.null_block: bool = null_block
        self.raw: Tensor = raw
        self.rope: Tensor | None = rope

        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        row_bytes = math.prod(self.shape[2:]) * self.dtype.itemsize
        self.row_bytes: int = row_bytes

        if not _is_valid_word_bank(self.raw) or self.dtype not in (
            torch.bfloat16,
            torch.float32,
            torch.uint8,
        ):
            raise ValueError("Invalid native cache bank or record dtype")

        raw_word_rows = _bank_word_rows(self.raw)
        rope_ok = (
            raw_word_rows % ROPE_COMPANION_DIVISOR == 0
            and self.rope is not None
            and _is_valid_word_bank(self.rope)
            and _bank_word_rows(self.rope) == raw_word_rows // ROPE_COMPANION_DIVISOR
        )
        if self.dtype == torch.uint8 and (row_bytes != ROW_BYTES or not rope_ok):
            raise ValueError("FP8 records are one pool row plus a RoPE companion")

        unaligned_record = row_bytes % WORD_BYTES != 0 or (
            row_bytes % ROW_BYTES != 0 and ROW_BYTES % row_bytes != 0
        )
        if (
            unaligned_record
            or stride_bytes % row_bytes != 0
            or offset_bytes % row_bytes != 0
            or self.raw.nbytes % row_bytes != 0
        ):
            raise ValueError("Cache pages and records must be word/row aligned")

        self.page_stride: int = stride_bytes // row_bytes
        self.row_offset: int = offset_bytes // row_bytes

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def device(self) -> torch.device:
        return self.raw.device

    def row_indices(self, slots: Tensor) -> Tensor:
        slots_i32 = slots.int()
        rows_per_page = self.shape[1]
        if self.page_stride == rows_per_page:
            base_records = slots_i32
        else:
            page_gap = self.page_stride - rows_per_page
            base_records = slots_i32 + (slots_i32 // rows_per_page) * page_gap
        if self.row_offset == 0:
            return base_records
        return self.row_offset + base_records

    def pages(self, block_table: Tensor) -> Tensor | None:
        rows_per_page = self.shape[1]
        min_block = 1 if self.null_block else 0
        raw_word_rows = _bank_word_rows(self.raw)
        misaligned_layout = (
            rows_per_page % PAGE_ROW_ALIGNMENT != 0
            or self.row_offset % rows_per_page != 0
            or self.page_stride % rows_per_page != 0
        )
        if misaligned_layout or raw_word_rows % rows_per_page != 0:
            return None
        page_scale = self.page_stride // rows_per_page
        page_offset = self.row_offset // rows_per_page
        mapped_pages = block_table.int() * page_scale + page_offset
        return torch.where(block_table < min_block, -1, mapped_pages)

    def __getitem__(self, indices: tuple[Tensor, Tensor]) -> Tensor:
        if self.dtype == torch.uint8:
            raise TypeError(
                "PagedCacheRows does not support direct FP8 row indexing; "
                "use dsv41_gather_sparse_fp8_kv on raw/rope banks"
            )
        block_ids = indices[0].int()
        row_ids = indices[1].int()
        min_block = 1 if self.null_block else 0
        invalid = (block_ids < min_block) | (row_ids < 0)
        flat_slots = block_ids.clamp(min=0) * self.shape[1] + row_ids.clamp(min=0)
        records = self.row_indices(flat_slots).masked_fill(invalid, -1)

        flat_records = records.reshape(-1)
        record_words = self.row_bytes // WORD_BYTES
        if record_words >= ROW_WORDS:
            query_rows = _expand_chunks(flat_records, record_words // ROW_WORDS)
            raw_words = cache_op.dsv41_gather_cache_rows(self.raw, query_rows)
            words = raw_words.reshape(-1, record_words)
        else:
            records_per_row = ROW_WORDS // record_words
            query_rows = torch.where(
                flat_records >= 0, flat_records // records_per_row, -1
            )
            raw_words = cache_op.dsv41_gather_cache_rows(self.raw, query_rows)
            batch_iota = torch.arange(flat_records.numel(), device=flat_records.device)
            sub_col = flat_records.clamp(min=0) % records_per_row
            words = raw_words.reshape(-1, records_per_row, record_words)[
                batch_iota, sub_col
            ]

        # Avoid dropping _last_read during torch.compile: XLA can otherwise
        # reorder donated in-place writes ahead of prior SWA/state gathers.
        if torch.compiler.is_compiling():
            self._last_read = raw_words.reshape(-1)[:1]
        if self.dtype == torch.float32:
            values = words.view(torch.float32)
        else:
            values = _unpack_bf16_record_words(words)
        return values.reshape(*records.shape, *self.shape[2:])

    @property
    def supports_paged_windows(self) -> bool:
        """Whether ``gather_paged_windows`` can serve this row-aligned bf16 pool."""
        record_words = self.row_bytes // WORD_BYTES
        return self.dtype == torch.bfloat16 and record_words % ROW_WORDS == 0

    def gather_paged_windows(
        self, block_table: Tensor, window_base: Tensor, window_records: int
    ) -> Tensor:
        """Gather ``[window_base, window_base + window_records)`` per table row.

        Shapes: block_table=[S, max_blocks], window_base=[S] (may be < 0) ->
        [S, window_records, *shape[2:]]; caller must mask null/unmapped slots.
        """
        rows_per_record = self.row_bytes // ROW_BYTES
        records_per_page = self.shape[1]
        window_rows = window_records * rows_per_record
        rows_per_page = records_per_page * rows_per_record
        # Avoid record-granular ceiling: kernel validates arbitrary row phases.
        pages_per_window = -(-(window_rows - 1) // rows_per_page) + 1
        base = window_base.to(torch.int64)
        first_col = torch.div(base, records_per_page, rounding_mode="floor")
        cols = first_col[:, None] + torch.arange(pages_per_window, device=base.device)
        table_width = block_table.shape[1]
        blocks = block_table.to(torch.int64).gather(1, cols.clamp(0, table_width - 1))
        min_block = 1 if self.null_block else 0
        mapped = (cols >= 0) & (cols < table_width) & (blocks >= min_block)
        page_ids = torch.where(mapped, blocks, -1).int()
        phase_rows = ((base - first_col * records_per_page) * rows_per_record).int()
        # Avoid self.raw.reshape(-1, ROW_WORDS): passing 3D root pools directly
        # prevents ExecuteCompiledModelImpl tt_jit_execute_as_strided copies.
        words = swa_window_op.dsv41_gather_swa_windows(
            self.raw,
            page_ids,
            phase_rows,
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=self.page_stride * rows_per_record,
            page_row_offset=self.row_offset * rows_per_record,
        )
        if torch.compiler.is_compiling():
            self._last_read = words.reshape(-1)[:1]
        record_words = self.row_bytes // WORD_BYTES
        values = _unpack_bf16_record_words(words.reshape(-1, record_words))
        return values.reshape(base.shape[0], window_records, *self.shape[2:])

    def _write_words_in_place(
        self,
        bank: Tensor,
        records: Tensor,
        words: Tensor,
        dependency: Tensor | None = None,
    ) -> None:
        page_tokens = self.shape[1]
        if words.shape[1] > ROW_WORDS:
            chunks_per_record = words.shape[1] // ROW_WORDS
            records = _expand_chunks(records, chunks_per_record)
            words = words.reshape(-1, ROW_WORDS)
            page_tokens *= chunks_per_record
        bank.copy_(
            cache_op.dsv41_write_sub_rows(
                bank,
                words,
                records,
                after=dependency,
                page_tokens=page_tokens,
            )
        )

    def write(
        self,
        slots: Tensor,
        values: Tensor,
        *,
        records: Tensor | None = None,
        valid: Tensor | None = None,
        after_read: bool = False,
    ) -> None:
        last_read = self._last_read
        self._last_read = None
        dependency = (
            last_read if (after_read and torch.compiler.is_compiling()) else None
        )
        if records is None:
            min_slot = self.shape[1] if self.null_block else 0
            invalid = slots < min_slot if valid is None else (slots < min_slot) | ~valid
            records = self.row_indices(slots.clamp(min=0)).masked_fill(invalid, -1)
        record_ids = records.int()
        if self.dtype == torch.uint8:
            assert self.rope is not None
            kv_words, rope_words = cache_op.dsv41_pack_fp8_kv_rows(
                values.to(torch.bfloat16)
            )
            self._write_words_in_place(self.raw, record_ids, kv_words, dependency)
            self._write_words_in_place(self.rope, record_ids, rope_words, dependency)
            return
        flat_values = values.to(self.dtype).reshape(values.shape[0], -1)
        if self.dtype == torch.float32:
            words = flat_values.view(torch.int32)
        else:
            even_bits = (
                flat_values[:, 0::2].float().view(torch.int32) >> BF16_SHIFT_BITS
            ) & BF16_EVEN_MASK
            odd_bits = flat_values[:, 1::2].float().view(torch.int32) & BF16_ODD_MASK
            words = even_bits | odd_bits
        self._write_words_in_place(self.raw, record_ids, words, dependency)


class DsV41CacheMixin(DsV41CacheLayer):
    """Lightweight cache-binding mixin for DeepSeek-V4.1 layers."""

    null_block: bool = False
    _bound_cache: Any = None
    _tpu_raw_cache: Any = None
    _tpu_rope_pool: Any = None
    _tpu_cache_layout: Any = None
    _tpu_paged_cache_rows: PagedCacheRows | None = None

    @property
    def paged_cache_rows(self) -> PagedCacheRows | None:
        return self._tpu_paged_cache_rows

    @property
    def kv_cache(self) -> Any:
        """Return ``PagedCacheRows`` under ``torch.compile`` or bound pools eagerly."""
        if torch.compiler.is_compiling() and self._tpu_paged_cache_rows is not None:
            return self._tpu_paged_cache_rows
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache: Any) -> None:
        if cache is not self._bound_cache:
            self._bound_cache = cache
            self._tpu_raw_cache = None
            self._tpu_rope_pool = None
            self._tpu_cache_layout = None
            self._tpu_paged_cache_rows = None
            self.null_block = False

    def bind_kv_cache(
        self,
        kv_cache: Any,
        *,
        raw_cache: RawCachePool | None = None,
        layout: tuple[tuple[int, ...], torch.dtype, int, int] | None = None,
        rope_pool: RawCachePool | None = None,
    ) -> None:
        self._bound_cache = kv_cache
        self._tpu_raw_cache = raw_cache
        self._tpu_rope_pool = rope_pool
        if raw_cache is None:
            self._tpu_cache_layout = layout
            self._tpu_paged_cache_rows = None
            self.null_block = False
            return
        if layout is None:
            assert isinstance(kv_cache, Tensor)
            elem_size = kv_cache.element_size()
            resolved_layout = (
                tuple(kv_cache.shape),
                kv_cache.dtype,
                kv_cache.stride(0) * elem_size,
                int(kv_cache.storage_offset()) * elem_size,
            )
        else:
            resolved_layout = layout
        self._tpu_cache_layout = resolved_layout
        self.null_block = True
        self._tpu_paged_cache_rows = PagedCacheRows(
            raw_cache,
            resolved_layout,
            rope_pool,
            null_block=True,
        )

    def zero_blocks(self, block_ids: list[int]) -> None:
        pcr = self._tpu_paged_cache_rows
        if pcr is None:
            bound = self._bound_cache
            if isinstance(bound, Tensor) and bound.ndim >= 1:
                valid = [b for b in block_ids if 0 < b < int(bound.shape[0])]
                if valid:
                    idx_t = torch.tensor(valid, dtype=torch.long, device=bound.device)
                    bound[idx_t] = 0
            return

        num_blocks = pcr.shape[0]
        valid = [b for b in block_ids if 0 < b < num_blocks]
        if not valid:
            return
        assert self._tpu_cache_layout is not None
        stride_rows = int(self._tpu_cache_layout[2]) // ROW_BYTES
        for bank, page_rows in (
            (pcr.raw, stride_rows),
            (pcr.rope, max(1, stride_rows // ROPE_COMPANION_DIVISOR)),
        ):
            if bank is None:
                continue
            bank_2d = bank.view(-1, ROW_WORDS)
            n_rows = int(bank_2d.shape[0])
            for bid in valid:
                start = bid * page_rows
                end = start + page_rows
                if end <= n_rows:
                    bank_2d[start:end] = 0


def write_cache(
    cache: PagedCacheRows | Tensor,
    slots: Tensor,
    values: Tensor,
    valid: Tensor,
    after_read: bool = False,
    *,
    records: Tensor | None = None,
) -> None:
    num_rows = values.shape[0]
    valid_mask = valid[:num_rows]
    if isinstance(cache, PagedCacheRows):
        masked_records = (
            None if records is None else records[:num_rows].masked_fill(~valid_mask, -1)
        )
        cache.write(
            slots[:num_rows],
            values,
            records=masked_records,
            valid=valid_mask,
            after_read=after_read,
        )
        return
    valid_slots = slots[:num_rows].masked_fill(~valid_mask, -1)[valid_mask].long()
    if cache.ndim == values.ndim:
        cache[(valid_slots,)] = values[valid_mask]
    else:
        rows_per_page = cache.shape[1]
        cache[valid_slots // rows_per_page, valid_slots % rows_per_page] = values[
            valid_mask
        ]


__all__ = [
    "ROPE_COMPANION_DIVISOR",
    "ROPE_WORDS",
    "ROW_BYTES",
    "ROW_WORDS",
    "WORD_BYTES",
    "DsV41CacheMixin",
    "PagedCacheRows",
    "RawCachePool",
    "write_cache",
]
