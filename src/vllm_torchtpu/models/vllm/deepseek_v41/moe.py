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
"""Self-contained DeepSeek-V4.1 MoE block with non-SP EP router gate slicing."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index


class _DeepseekV41SharedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn: nn.Module = (
            SiluAndMulWithClamp(swiglu_limit)
            if swiglu_limit is not None
            else SiluAndMul()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV41MoE(nn.Module):
    """Self-contained DeepSeek-V4.1 MoE block with non-SP EP gate input slicing."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "layers.0.ffn",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("V4.1 TPU MoE does not support EPLB")
        if vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS:
            raise NotImplementedError("V4.1 TPU requires the native V4 MoE backend")

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.use_sequence_parallel = use_sequence_parallel
        self.reduce_results = reduce_results
        self.use_native_mega_moe: bool = False
        self.n_redundant_experts: int = 0

        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        # Avoid direct access on optional fields: DeepseekV41Config flattens
        # text_config kwargs without setting defaults on the config class.
        if extract_layer_index(prefix) >= int(config.num_hidden_layers):
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )

        if n_routed_experts % self.tp_size != 0:
            raise ValueError(
                f"n_routed_experts ({n_routed_experts}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size
        self.n_routed_experts = n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts: nn.Module | None = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = _DeepseekV41SharedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=n_routed_experts,
            top_k=n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=cast(
                torch.Tensor | None, self.gate.e_score_correction_bias
            ),
            swiglu_limit=self.swiglu_limit,
            reduce_results=reduce_results,
            router_logits_dtype=torch.float32,
            apply_routed_scale_to_output=True,
        )
        # Avoid duplicate gate evaluation inside MoERunner after rank slicing.
        self.experts.gate = None

    def _maybe_slice_gate_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_config = self.experts.moe_config
        routed_experts = self.experts.routed_experts
        quant_method = routed_experts.quant_method
        group = get_tp_group() if moe_config.dp_size > 1 else get_ep_group()
        num_tokens = hidden_states.shape[0]
        world_size = group.world_size
        use_ep = moe_config.moe_parallel_config.use_ep and quant_method.is_monolithic
        has_custom_routing = (
            quant_method.supports_internal_mk
            or routed_experts.custom_routing_function is not None
        )
        if (
            self.use_sequence_parallel
            or moe_config.pcp_size > 1
            or not use_ep
            or has_custom_routing
            or world_size <= 1
        ):
            return hidden_states
        if num_tokens % world_size != 0:
            raise ValueError(
                f"Tokens ({num_tokens}) must be divisible by group size ({world_size})"
            )
        chunk_tokens = num_tokens // world_size
        start = group.rank_in_group * chunk_tokens
        return hidden_states[start : start + chunk_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_input = self._maybe_slice_gate_input(hidden_states)
        router_logits, _ = self.gate(gate_input)
        expert_output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        return expert_output.view(hidden_states.shape)
