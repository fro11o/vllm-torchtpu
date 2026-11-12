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
"""Host planning and DP lockstep helpers for Context-Encoder-Decoder (CED) replay."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.config import VllmConfig


def prefix_replay_enabled(vllm_config: VllmConfig, *, ced_enabled: bool = True) -> bool:
    """Return whether CED runs with prefix caching through SWA bounded replay."""
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    replay_ok = (
        ced_enabled
        and cache_config.enable_prefix_caching
        and cache_config.swa_bounded_replay
    )
    return bool(replay_ok and parallel_config.prefill_context_parallel_size == 1)


@dataclass(frozen=True)
class DecoderReplayPlan:
    """Host-side token narrowing and bucket plan for the CED decoder stage.

    Shapes: query_start_loc=[num_reqs + 1], replay_starts=[num_reqs].
    """

    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int
    is_identity: bool = False


def _as_int_sequence(values: Sequence[int] | torch.Tensor) -> Sequence[int]:
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return values


def plan_decoder_replay(
    metadata: Any,
    window: int,
    num_tokens: int,
    compile_sizes: Iterable[int],
) -> DecoderReplayPlan | None:
    """Compute the prompt-tail SWA replay plan across scheduled chunks and decode."""
    query_start_loc = metadata.query_start_loc_cpu
    seq_lens = metadata.seq_lens_cpu
    prompt_lens = metadata.prompt_lens_cpu
    if query_start_loc is None or seq_lens is None or prompt_lens is None:
        return None

    query_bounds = _as_int_sequence(query_start_loc)
    seq_lengths = _as_int_sequence(seq_lens)
    prompt_lengths = _as_int_sequence(prompt_lens)
    replay_starts_cpu = metadata.replay_starts_cpu
    num_reqs = len(seq_lengths)
    replay_hits = (
        (0,) * num_reqs
        if replay_starts_cpu is None
        else _as_int_sequence(replay_starts_cpu)
    )
    invalid_lengths = (
        len(query_bounds),
        len(prompt_lengths),
        len(replay_hits),
    ) != (num_reqs + 1, num_reqs, num_reqs)
    if (
        window < 1
        or invalid_lengths
        or query_bounds[0] != 0
        or query_bounds[-1] > num_tokens
    ):
        raise ValueError("Invalid decoder replay scheduling metadata")

    replay_starts: list[int] = []
    kept_indices: list[int] = []
    out_query_bounds: list[int] = [0]
    for start, end, seq_len, prompt_len, hit_start in zip(
        query_bounds,
        query_bounds[1:],
        seq_lengths,
        prompt_lengths,
        replay_hits,
    ):
        if end < start or seq_len < end - start or prompt_len < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        prompt_tail_start = prompt_len - window if prompt_len > window else 0
        replay_start = max(prompt_tail_start, hit_start)
        replay_starts.append(replay_start)
        chunk_start_pos = seq_len - (end - start)
        first_kept_offset = max(0, replay_start - chunk_start_pos)
        kept_indices.extend(range(start + first_kept_offset, end))
        out_query_bounds.append(len(kept_indices))

    kept_count = len(kept_indices)
    candidate_buckets = (
        int(bucket)
        for bucket in compile_sizes
        if kept_count <= int(bucket) < num_tokens
    )
    padded_num_tokens = (
        0 if kept_count == 0 else min(candidate_buckets, default=num_tokens)
    )
    is_identity = (
        kept_count > 0
        and kept_count == query_bounds[-1]
        and padded_num_tokens == num_tokens
    )
    return DecoderReplayPlan(
        indices=tuple(kept_indices),
        query_start_loc=tuple(out_query_bounds),
        replay_starts=tuple(replay_starts),
        padded_num_tokens=padded_num_tokens,
        is_identity=is_identity,
    )


def sync_decoder_replay_plan(
    plan: DecoderReplayPlan | None,
    metadata: Any,
    num_tokens: int,
    dp_cpu_group: Any = None,
    *,
    window: int | None = None,
    prefix_replay: bool = False,
    compile_sizes: tuple[int, ...] = (),
) -> DecoderReplayPlan | None:
    """Synchronize decoder stage execution and padded token bucket across DP ranks.

    Precondition: ``num_tokens`` is DP-uniform across ``dp_cpu_group`` ranks.
    """
    if dp_cpu_group is None:
        return plan
    min_compile_bucket = min((size for size in compile_sizes if size > 0), default=0)
    short_batch_limit = (
        0 if not window or window <= 0 else max(window, min_compile_bucket)
    )
    # Avoid a host Gloo all_reduce on every short decode step: when DP-uniform
    # num_tokens <= max(window, min_compile_bucket) without prefix replay, no
    # smaller compile bucket exists so every rank agrees on target_tokens=num_tokens.
    if not prefix_replay and 0 < num_tokens <= short_batch_limit:
        target_tokens = num_tokens
    else:
        local_vals = [
            1 if plan is not None else 0,
            plan.padded_num_tokens if plan is not None else 0,
        ]
        state = torch.tensor(local_vals, dtype=torch.int32, device="cpu")
        dist.all_reduce(state, op=dist.ReduceOp.MAX, group=dp_cpu_group)
        if state[0].item() == 0:
            return None
        target_tokens = int(state[1].item())

    if plan is None:
        num_reqs = int(metadata.seq_lens.shape[0])
        return DecoderReplayPlan(
            indices=(),
            query_start_loc=(0,) * (num_reqs + 1),
            replay_starts=(0,) * num_reqs,
            padded_num_tokens=target_tokens,
            is_identity=False,
        )
    if plan.padded_num_tokens == target_tokens:
        return plan
    query_bounds = metadata.query_start_loc_cpu
    kept_count = len(plan.indices)
    is_identity = (
        kept_count > 0
        and query_bounds is not None
        and kept_count == query_bounds[-1]
        and target_tokens == num_tokens
    )
    return replace(plan, padded_num_tokens=target_tokens, is_identity=is_identity)


def build_decoder_inverse_rows(
    indices: tuple[int, ...],
    padded: int,
    total: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the conflict-free ``int32[total]`` gather map for decoder outputs."""
    kept_row_by_encoder_row = {
        encoder_row: kept_pos for kept_pos, encoder_row in enumerate(indices)
    }
    gather_indices = [kept_row_by_encoder_row.get(row, padded) for row in range(total)]
    return torch.tensor(gather_indices, dtype=torch.int32, device=device)


def scatter_decoder_outputs(
    tensor: torch.Tensor, inverse_rows: torch.Tensor
) -> torch.Tensor:
    """Restore narrowed decoder rows ``[padded, ...]`` into ``[total, ...]``."""
    if tensor.ndim < 1:
        raise ValueError("scatter_decoder_outputs requires at least a 1D tensor")
    # Avoid out.index_copy_ with duplicate sink indices (lowers to conflicting
    # XlaScatter) and fixed 4-tuple F.pad (pads dim -2 instead of dim 0 for ndim != 2).
    pad_spec = (0, 0) * (tensor.ndim - 1) + (0, 1)
    return F.pad(tensor, pad_spec).index_select(0, inverse_rows)
