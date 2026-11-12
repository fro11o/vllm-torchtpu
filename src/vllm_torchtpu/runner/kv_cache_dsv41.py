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
"""DeepSeek-V4.1 single-bank word-pool KV-cache planning, allocation, and overlay."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import Tensor
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

type RawCachePool = Tensor

ROW_WORDS: int = 128
ROPE_WORDS: int = 32
WORD_BYTES: int = torch.int32.itemsize
ROW_BYTES: int = ROW_WORDS * WORD_BYTES
ROPE_COMPANION_DIVISOR: int = ROW_WORDS // ROPE_WORDS
SPARSECORE_MAX_TENSOR_BYTES: int = 16 << 30

_INDEXER_SUFFIX: str = ".indexer.k_cache"
_MIN_MOSAIC_SUBLANES: int = 8
_MIN_MLA_STATES_FOR_ROPE_SUBLANES: int = _MIN_MOSAIC_SUBLANES * ROPE_COMPANION_DIVISOR
_TARGET_HOST_ROWS: int = 64
_MAX_AUTO_MLA_BLOCK_SIZE: int = 256


class DsV41CacheLayer(nn.Module):
    """Base marker class for DeepSeek-V4.1 layers bound to word-pool caches."""

    def bind_kv_cache(
        self,
        kv_cache: Any,
        *,
        raw_cache: RawCachePool | None = None,
        layout: tuple[tuple[int, ...], torch.dtype, int, int] | None = None,
        rope_pool: RawCachePool | None = None,
    ) -> None:
        raise NotImplementedError


class DsV41BlockZeroOwner(Protocol):
    """Protocol for runner block-zeroing delegation when DSv4.1 pools are active."""

    def zero_blocks(
        self,
        block_ids: list[int],
        *,
        split: int = 1,
        copy_blocks_fn: Callable[[list[Tensor], Tensor, Tensor], Any] | None = None,
    ) -> None: ...


class DsV41RunnerHost(Protocol):
    """Structural runner contract required by ``DsV41KVCacheAllocator``."""

    kv_cache_raw_tensors: list[Tensor]
    ds_v41_zero_owner: DsV41BlockZeroOwner | None

    @property
    def device(self) -> torch.device: ...

    @property
    def enforce_eager(self) -> bool: ...

    @property
    def kv_cache_config(self) -> KVCacheConfig | None: ...

    @property
    def max_num_reqs(self) -> int: ...

    @property
    def shared_kv_cache_layers(self) -> Mapping[str, str]: ...

    def _precompile_mamba_state_seed_copies(self) -> None: ...


@dataclass(frozen=True)
class _PendingLayerBinding:
    bound_cache: Tensor | tuple[Tensor, Tensor]
    raw_pool: RawCachePool
    layout: tuple[tuple[int, ...], torch.dtype, int, int]
    rope_pool: RawCachePool | None = None


def is_cache_for_ds_v41(module: object) -> bool:
    return isinstance(module, DsV41CacheLayer)


def _largest_power_of_two_at_most(limit: int) -> int:
    return 1 << (max(1, limit).bit_length() - 1)


def _smallest_power_of_two_at_least(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _mla_nope_rows(spec: MLAAttentionSpec, block_size: int | None = None) -> int:
    bs = spec.block_size if block_size is None else block_size
    num_states = bs // int(spec.tokens_per_state)
    return (num_states * spec.head_size) // ROW_BYTES


def _indexer_rows(spec: MLAAttentionSpec, block_size: int | None = None) -> int:
    bs = spec.block_size if block_size is None else block_size
    num_states = bs // int(spec.tokens_per_state)
    return (num_states * spec.state_content_size_bytes) // ROW_BYTES


def _pack_homogeneous_bins(
    items: list[tuple[str, int]],
    max_pool_rows: int,
) -> list[tuple[int, list[tuple[str, int, int]]]]:
    by_rows: dict[int, list[str]] = defaultdict(list)
    for name, item_rows in items:
        by_rows[item_rows].append(name)

    bins: list[tuple[int, list[tuple[str, int, int]]]] = []
    for item_rows in sorted(by_rows.keys(), reverse=True):
        names = by_rows[item_rows]
        max_items = max(1, max_pool_rows // item_rows)
        cursor = 0
        while cursor < len(names):
            remaining = len(names) - cursor
            chunk_count = _largest_power_of_two_at_most(min(remaining, max_items))
            placements = [
                (names[cursor + idx], item_rows, idx * item_rows)
                for idx in range(chunk_count)
            ]
            bins.append((chunk_count * item_rows, placements))
            cursor += chunk_count
    return bins


def _pack_dyadic_bins(
    items: list[tuple[str, int]],
    max_pool_rows: int,
) -> list[tuple[int, list[tuple[str, int, int]]]]:
    buckets: dict[int, list[list[tuple[str, int, int]]]] = defaultdict(list)
    for name, item_rows in items:
        buckets[item_rows].append([(name, item_rows, 0)])

    while True:
        mergeable = [
            size
            for size, entries in buckets.items()
            if len(entries) >= 2 and size * 2 <= max_pool_rows
        ]
        if not mergeable:
            break
        size = min(mergeable)
        entries = buckets[size]
        while len(entries) >= 2:
            left = entries.pop(0)
            right = entries.pop(0)
            shifted_right = [
                (name, rows, offset + size) for name, rows, offset in right
            ]
            buckets[size * 2].append([*left, *shifted_right])

    bins: list[tuple[int, list[tuple[str, int, int]]]] = []
    for bin_rows in sorted(buckets.keys(), reverse=True):
        for placements in buckets[bin_rows]:
            bins.append((bin_rows, placements))
    return bins


def _slots_from_hosts(host_rows_list: list[int], req_rows: int) -> int:
    return sum(
        rows // req_rows
        for rows in host_rows_list
        if rows >= req_rows and rows % req_rows == 0
    )


def _collect_overlay_slots(
    non_rope_hosts: list[Tensor],
    req_rows: int,
) -> list[tuple[Tensor, int, int]]:
    slots: list[tuple[Tensor, int, int]] = []
    for host in non_rope_hosts:
        host_rows = int(host.shape[1])
        if host_rows >= req_rows and host_rows % req_rows == 0:
            for sub_idx in range(host_rows // req_rows):
                slots.append((host, host_rows, sub_idx * req_rows))
    return slots


def _conservative_overlay_slots(
    mla_specs: list[tuple[str, MLAAttentionSpec]],
    idx_specs: list[tuple[str, MLAAttentionSpec]],
    mla_block_size: int,
    tc_bin_cap: int,
    req_rows: int,
) -> int:
    if req_rows <= 0:
        return 0
    idx_items = [
        (name, _indexer_rows(spec, mla_block_size)) for name, spec in idx_specs
    ]
    idx_rows = [bin_rows for bin_rows, _ in _pack_dyadic_bins(idx_items, tc_bin_cap)]
    mla_items = [
        (name, _mla_nope_rows(spec, mla_block_size)) for name, spec in mla_specs
    ]
    merged_mla = [
        bin_rows for bin_rows, _ in _pack_homogeneous_bins(mla_items, _TARGET_HOST_ROWS)
    ]
    # Avoid assuming MLA bins always merge to _TARGET_HOST_ROWS: large block counts
    # can clamp sc_bin_cap below 64, leaving each MLA layer in its own pool.
    unmerged_mla = [nope_rows for _, nope_rows in mla_items]
    return min(
        _slots_from_hosts([*merged_mla, *idx_rows], req_rows),
        _slots_from_hosts([*unmerged_mla, *idx_rows], req_rows),
    )


def _pad_group_specs_to_slot_cap(
    updated: dict[str, KVCacheSpec],
    specs: list[tuple[str, Any]],
    slots: int,
    anchor_bytes: int,
    target_block_bytes: int,
) -> None:
    if not specs:
        return
    unpadded_bytes = specs[0][1].real_page_size_bytes
    if slots <= 0:
        for name, spec in specs:
            updated[name] = dataclasses.replace(
                spec, page_size_padded=max(target_block_bytes, unpadded_bytes)
            )
        return
    if len(specs) > slots and anchor_bytes // unpadded_bytes > slots:
        min_padded_bytes = (anchor_bytes // (slots + 1)) + 1
        padded_bytes = min(
            anchor_bytes,
            ((min_padded_bytes + ROW_BYTES - 1) // ROW_BYTES) * ROW_BYTES,
        )
        for name, spec in specs:
            updated[name] = dataclasses.replace(spec, page_size_padded=padded_bytes)


class DsV41KVCacheAllocator:
    """Plans and materializes single-bank ``int32[N, R, 128]`` KV pools for DSv4.1."""

    def __init__(self, runner: DsV41RunnerHost) -> None:
        self.runner = runner
        self._pending_bindings: dict[str, _PendingLayerBinding] = {}

    @staticmethod
    def prepare_kv_cache_specs(
        kv_cache_specs: dict[str, KVCacheSpec],
        default_block_size: int,
    ) -> dict[str, KVCacheSpec]:
        mla_specs: list[tuple[str, MLAAttentionSpec]] = []
        idx_specs: list[tuple[str, MLAAttentionSpec]] = []
        swa_specs: list[tuple[str, SlidingWindowMLASpec]] = []
        state_specs: list[tuple[str, CircularBufferSpec]] = []

        for name, spec in kv_cache_specs.items():
            if isinstance(spec, SlidingWindowMLASpec):
                swa_specs.append((name, spec))
            elif isinstance(spec, MLAAttentionSpec):
                if name.endswith(_INDEXER_SUFFIX):
                    idx_specs.append((name, spec))
                else:
                    mla_specs.append((name, spec))
            elif isinstance(spec, CircularBufferSpec):
                state_specs.append((name, spec))

        if not mla_specs and not idx_specs:
            return kv_cache_specs

        lcm_ratio = math.lcm(
            *(int(spec.tokens_per_state) for _, spec in (*mla_specs, *idx_specs))
        )
        # Avoid mla_block_size % (32 * lcm_ratio) != 0: compressed RoPE (1/4 row)
        # and Indexer (1/2 row) pages must stay >= TPU7x's 8-sublane DMA minimum.
        min_mla_bs = _MIN_MLA_STATES_FOR_ROPE_SUBLANES * lcm_ratio
        mla_bs = math.lcm(max(1, default_block_size), min_mla_bs)

        swa_rows = (
            max(spec.real_page_size_bytes // ROW_BYTES for _, spec in swa_specs)
            if swa_specs
            else 0
        )
        state_rows = (
            max(spec.real_page_size_bytes // ROW_BYTES for _, spec in state_specs)
            if state_specs
            else 0
        )
        tc_bin_cap = _smallest_power_of_two_at_least(
            max(_TARGET_HOST_ROWS, swa_rows, state_rows)
        )

        while mla_bs * 2 <= _MAX_AUTO_MLA_BLOCK_SIZE:
            swa_ok = (
                swa_rows == 0
                or _conservative_overlay_slots(
                    mla_specs, idx_specs, mla_bs, tc_bin_cap, swa_rows
                )
                > 0
            )
            state_ok = (
                state_rows == 0
                or _conservative_overlay_slots(
                    mla_specs, idx_specs, mla_bs, tc_bin_cap, state_rows
                )
                > 0
            )
            if swa_ok and state_ok:
                break
            mla_bs *= 2

        updated = dict(kv_cache_specs)
        for name, spec in (*mla_specs, *idx_specs):
            updated[name] = dataclasses.replace(
                spec, block_size=mla_bs, page_size_padded=None
            )

        anchor_bytes = sum(
            updated[name].page_size_bytes for name, _ in (*mla_specs, *idx_specs)
        )
        swa_slots = _conservative_overlay_slots(
            mla_specs, idx_specs, mla_bs, tc_bin_cap, swa_rows
        )
        state_slots = _conservative_overlay_slots(
            mla_specs, idx_specs, mla_bs, tc_bin_cap, state_rows
        )
        extra_swa_rows = swa_rows if (swa_rows > 0 and swa_slots == 0) else 0
        if (
            state_rows > 0
            and state_slots == 0
            and extra_swa_rows >= state_rows
            and extra_swa_rows % state_rows == 0
        ):
            state_slots = extra_swa_rows // state_rows
        extra_state_rows = state_rows if (state_rows > 0 and state_slots == 0) else 0
        target_block_bytes = (
            anchor_bytes + (extra_swa_rows + extra_state_rows) * ROW_BYTES
        )

        # Avoid > available slots per BLHNC group: excess same-group SWA/State
        # layers cannot share host pools and would allocate unbudgeted HBM.
        _pad_group_specs_to_slot_cap(
            updated, swa_specs, swa_slots, anchor_bytes, target_block_bytes
        )
        _pad_group_specs_to_slot_cap(
            updated, state_specs, state_slots, anchor_bytes, target_block_bytes
        )

        return updated

    def _classify_layers(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
    ) -> tuple[
        list[tuple[str, MLAAttentionSpec]],
        list[tuple[str, MLAAttentionSpec]],
        list[list[tuple[str, SlidingWindowMLASpec]]],
        list[list[tuple[str, CircularBufferSpec]]],
    ]:
        mla_layers: list[tuple[str, MLAAttentionSpec]] = []
        idx_layers: list[tuple[str, MLAAttentionSpec]] = []
        swa_groups: list[list[tuple[str, SlidingWindowMLASpec]]] = []
        state_groups: list[list[tuple[str, CircularBufferSpec]]] = []

        for group in kv_cache_config.kv_cache_groups:
            swa_in_group: list[tuple[str, SlidingWindowMLASpec]] = []
            state_in_group: list[tuple[str, CircularBufferSpec]] = []
            for name in group.layer_names:
                if name in self.runner.shared_kv_cache_layers:
                    continue
                spec = per_layer_spec(name)
                if isinstance(spec, SlidingWindowMLASpec):
                    swa_in_group.append((name, spec))
                elif isinstance(spec, MLAAttentionSpec):
                    if name.endswith(_INDEXER_SUFFIX):
                        idx_layers.append((name, spec))
                    else:
                        mla_layers.append((name, spec))
                elif isinstance(spec, CircularBufferSpec):
                    state_in_group.append((name, spec))
                else:
                    raise ValueError(
                        f"Unsupported DeepSeek-V4.1 KV cache layer {name!r}: {spec!r}"
                    )
            if swa_in_group:
                swa_groups.append(swa_in_group)
            if state_in_group:
                state_groups.append(state_in_group)

        return mla_layers, idx_layers, swa_groups, state_groups

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, Any],
    ) -> None:
        self._pending_bindings.clear()
        num_blocks = kv_cache_config.num_blocks
        device = self.runner.device
        mla_layers, idx_layers, swa_groups, state_groups = self._classify_layers(
            kv_cache_config, per_layer_spec
        )

        raw_tensors: list[Tensor] = []
        non_rope_hosts: list[Tensor] = []

        def _alloc_pool(pool_rows: int) -> Tensor:
            pool = torch.zeros(
                (num_blocks, pool_rows, ROW_WORDS), dtype=torch.int32, device=device
            )
            raw_tensors.append(pool)
            return pool

        max_sc_rows = max(
            _MIN_MOSAIC_SUBLANES,
            SPARSECORE_MAX_TENSOR_BYTES // (max(num_blocks, 1) * ROW_BYTES),
        )
        sc_bin_cap = min(_TARGET_HOST_ROWS, max_sc_rows)

        max_swa_rows = max(
            (
                spec.real_page_size_bytes // ROW_BYTES
                for group in swa_groups
                for _, spec in group
            ),
            default=0,
        )
        max_state_rows = max(
            (
                spec.real_page_size_bytes // ROW_BYTES
                for group in state_groups
                for _, spec in group
            ),
            default=0,
        )
        tc_bin_cap = _smallest_power_of_two_at_least(
            max(_TARGET_HOST_ROWS, max_swa_rows, max_state_rows)
        )

        mla_packable: list[tuple[str, int]] = []
        mla_spec_by_name = dict(mla_layers)
        for name, spec in mla_layers:
            nope_rows = _mla_nope_rows(spec)
            if nope_rows > max_sc_rows:
                req_gib = (num_blocks * nope_rows * ROW_BYTES) / (1 << 30)
                raise ValueError(
                    f"MLA layer {name!r} requires {req_gib:.2f} GiB "
                    f"({num_blocks} blocks x {nope_rows} rows), exceeding the "
                    f"16 GiB single-bank SparseCore limit (max_sc_rows={max_sc_rows})"
                )
            mla_packable.append((name, nope_rows))

        mla_bins = _pack_homogeneous_bins(mla_packable, sc_bin_cap)
        idx_spec_by_name = dict(idx_layers)
        idx_items = [(name, _indexer_rows(spec)) for name, spec in idx_layers]
        idx_bins = _pack_dyadic_bins(idx_items, tc_bin_cap)

        # Avoid allocating RoPE before non-RoPE host pools: SWA and State groups
        # overlay onto non-RoPE pools sorted largest-first.
        pending_rope_allocs: list[tuple[int, Tensor, list[tuple[str, int, int]]]] = []
        for pool_rows, placements in mla_bins:
            nope_pool = _alloc_pool(pool_rows)
            non_rope_hosts.append(nope_pool)
            pending_rope_allocs.append((pool_rows, nope_pool, placements))

        for pool_rows, placements in idx_bins:
            idx_pool = _alloc_pool(pool_rows)
            non_rope_hosts.append(idx_pool)
            for name, _item_rows, row_offset in placements:
                spec = idx_spec_by_name[name]
                kv_caches[name] = idx_pool
                self._pending_bindings[name] = _PendingLayerBinding(
                    bound_cache=idx_pool,
                    raw_pool=idx_pool,
                    layout=(
                        (num_blocks, spec.num_states, spec.head_size),
                        torch.bfloat16,
                        pool_rows * ROW_BYTES,
                        row_offset * ROW_BYTES,
                    ),
                )

        # Avoid pairing heterogeneous RoPE layers into one pool: csa_gather shares
        # one physical page table across NoPE and RoPE banks.
        for pool_rows, nope_pool, placements in pending_rope_allocs:
            rope_pool = _alloc_pool(pool_rows // ROPE_COMPANION_DIVISOR)
            for name, item_rows, row_offset in placements:
                spec = mla_spec_by_name[name]
                assert item_rows == _mla_nope_rows(spec)
                bound_pair = (nope_pool, rope_pool)
                kv_caches[name] = bound_pair
                self._pending_bindings[name] = _PendingLayerBinding(
                    bound_cache=bound_pair,
                    raw_pool=nope_pool,
                    layout=(
                        (num_blocks, spec.num_states, spec.head_size),
                        torch.uint8,
                        pool_rows * ROW_BYTES,
                        row_offset * ROW_BYTES,
                    ),
                    rope_pool=rope_pool,
                )

        non_rope_hosts.sort(key=lambda t: t.shape[1], reverse=True)

        if swa_groups:
            first_swa_spec = swa_groups[0][0][1]
            swa_rows = first_swa_spec.real_page_size_bytes // ROW_BYTES
            swa_slots = _collect_overlay_slots(non_rope_hosts, swa_rows)

            for group_layers in swa_groups:
                for pos, (name, spec) in enumerate(group_layers):
                    page_rows = spec.real_page_size_bytes // ROW_BYTES
                    if page_rows == swa_rows:
                        while pos >= len(swa_slots):
                            overflow = _alloc_pool(page_rows)
                            non_rope_hosts.append(overflow)
                            swa_slots.append((overflow, page_rows, 0))
                        raw_host, host_rows, row_offset = swa_slots[pos]
                    else:
                        overflow = _alloc_pool(page_rows)
                        non_rope_hosts.append(overflow)
                        raw_host, host_rows, row_offset = overflow, page_rows, 0

                    # Expose 4D int32[N, *, block_size, 128] so NgramHashState
                    # reads shape[2] == spec.block_size and shape[0] == num_blocks.
                    view_rows = int(raw_host.shape[1])
                    swa_view = raw_host.view(
                        num_blocks,
                        max(1, view_rows // spec.block_size),
                        spec.block_size,
                        (view_rows * ROW_WORDS)
                        // (max(1, view_rows // spec.block_size) * spec.block_size),
                    )
                    kv_caches[name] = swa_view
                    self._pending_bindings[name] = _PendingLayerBinding(
                        bound_cache=swa_view,
                        raw_pool=raw_host,
                        layout=(
                            (num_blocks, spec.block_size, spec.head_size),
                            torch.bfloat16,
                            host_rows * ROW_BYTES,
                            row_offset * ROW_BYTES,
                        ),
                    )

        if state_groups:
            first_state_spec = state_groups[0][0][1]
            state_rows = first_state_spec.real_page_size_bytes // ROW_BYTES
            state_slots = _collect_overlay_slots(non_rope_hosts, state_rows)

            for group_layers in state_groups:
                for pos, (name, spec) in enumerate(group_layers):
                    req_rows = spec.real_page_size_bytes // ROW_BYTES
                    if req_rows == state_rows:
                        while pos >= len(state_slots):
                            overflow = _alloc_pool(req_rows)
                            non_rope_hosts.append(overflow)
                            state_slots.append((overflow, req_rows, 0))
                        raw_host, host_rows, row_offset = state_slots[pos]
                    else:
                        overflow = _alloc_pool(req_rows)
                        non_rope_hosts.append(overflow)
                        raw_host, host_rows, row_offset = overflow, req_rows, 0

                    kv_caches[name] = raw_host
                    self._pending_bindings[name] = _PendingLayerBinding(
                        bound_cache=raw_host,
                        raw_pool=raw_host,
                        layout=(
                            (num_blocks, spec.block_size, spec.head_size),
                            spec.dtype,
                            host_rows * ROW_BYTES,
                            row_offset * ROW_BYTES,
                        ),
                    )

        self._validate_group_disjointness(kv_cache_config)
        self.runner.kv_cache_raw_tensors = raw_tensors
        self.runner.ds_v41_zero_owner = self
        if raw_tensors:
            synchronize_tensors(raw_tensors)
        if raw_tensors and not self.runner.enforce_eager:
            self.runner._precompile_mamba_state_seed_copies()  # noqa: SLF001 -- runner precompile hook

        logger.info(
            "DeepSeek-V4.1 KV cache: %d contiguous root pools (%s rows/block) "
            "for %d layers (mla=%d, indexer=%d, swa=%d in %d group(s), "
            "state=%d in %d group(s)), num_blocks=%d",
            len(raw_tensors),
            [int(t.shape[1]) for t in raw_tensors],
            len(kv_caches),
            len(mla_layers),
            len(idx_layers),
            sum(len(g) for g in swa_groups),
            len(swa_groups),
            sum(len(g) for g in state_groups),
            len(state_groups),
            num_blocks,
        )

    def bind_paged_caches(self, forward_context: Mapping[str, Any]) -> None:
        if not self._pending_bindings:
            return
        for name, target in self.runner.shared_kv_cache_layers.items():
            if target in self._pending_bindings and name not in self._pending_bindings:
                self._pending_bindings[name] = self._pending_bindings[target]

        for name, binding in self._pending_bindings.items():
            layer = forward_context.get(name)
            if isinstance(layer, DsV41CacheLayer):
                layer.bind_kv_cache(
                    binding.bound_cache,
                    raw_cache=binding.raw_pool,
                    layout=binding.layout,
                    rope_pool=binding.rope_pool,
                )

    def zero_blocks(
        self,
        block_ids: list[int],
        *,
        split: int = 1,
        copy_blocks_fn: Callable[[list[Tensor], Tensor, Tensor], Any] | None = None,
    ) -> None:
        raws = self.runner.kv_cache_raw_tensors
        if not block_ids or not raws:
            return
        num_pool_blocks = int(raws[0].shape[0])
        dst_ids = [
            bid * split + sub
            for bid in block_ids
            if bid > 0
            for sub in range(split)
            if bid * split + sub < num_pool_blocks
        ]
        if not dst_ids:
            return
        if raws[0].device.type == "cpu":
            idx = torch.tensor(dst_ids, dtype=torch.long)
            for raw in raws:
                if int(raw[0].abs().max().item()) != 0:
                    raise RuntimeError(
                        "DSv4.1 KV pool null_block=0 must remain all-zero for "
                        "null-block seed zeroing"
                    )
                raw[idx] = raw[0].clone()
            return
        if copy_blocks_fn is None:
            raise ValueError("copy_blocks_fn is required for non-CPU DSv4.1 pools")
        num_groups = (
            len(self.runner.kv_cache_config.kv_cache_groups)
            if self.runner.kv_cache_config is not None
            else 1
        )
        raw_cap = max(1, num_groups) * max(1, self.runner.max_num_reqs) * split
        chunk_cap = 8
        while chunk_cap < raw_cap:
            chunk_cap *= 4
        for offset in range(0, len(dst_ids), chunk_cap):
            sub_chunk = dst_ids[offset : offset + chunk_cap]
            bucket_len = 8
            while bucket_len < len(sub_chunk):
                bucket_len *= 4
            padded = sub_chunk + [0] * (bucket_len - len(sub_chunk))
            src_t = torch.zeros(len(padded), dtype=torch.int32).to(self.runner.device)
            dst_t = torch.tensor(padded, dtype=torch.int32).to(self.runner.device)
            copy_blocks_fn(raws, src_t, dst_t)

    def _validate_group_disjointness(self, kv_cache_config: KVCacheConfig) -> None:
        for group in kv_cache_config.kv_cache_groups:
            intervals_by_pool: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
            for name in group.layer_names:
                if name in self.runner.shared_kv_cache_layers:
                    continue
                binding = self._pending_bindings[name]
                shape, dtype, _, offset_bytes = binding.layout
                page_bytes = shape[1] * shape[2] * dtype.itemsize
                pool_id = id(binding.raw_pool)
                start = offset_bytes
                end = offset_bytes + page_bytes
                for prev_start, prev_end, prev_name in intervals_by_pool[pool_id]:
                    if start < prev_end and prev_start < end:
                        raise ValueError(
                            f"DeepSeek-V4.1 KV overlay collision in group "
                            f"{group.layer_names}: {name!r} [{start}, {end}) "
                            f"overlaps {prev_name!r} [{prev_start}, {prev_end})"
                        )
                intervals_by_pool[pool_id].append((start, end, name))
