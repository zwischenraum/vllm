# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for short-conv spec-decode state widening.

Speculative decoding verifies ``1 + num_spec`` query tokens per decode request
in a single forward. The short-conv rolling window must therefore reserve
``num_spec`` extra slots so it can roll back to the last accepted token
(mirrors ``mamba2``/``gated_delta_net``). Without the extra slots the captured
CUDA-graph decode silently corrupts generation on rejection.

This locks the pure shape contract; the layer forward + kernel path is covered
by the spec-decode e2e tests.
"""

from __future__ import annotations

import pytest

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

# conv_dim and (conv_kernel - 1) are kept distinct so ``set(shape)`` is an
# unambiguous, layout-agnostic view of the two axes.
INTERMEDIATE_SIZE = 128
CONV_KERNEL = 4  # -> baseline state_len = conv_kernel - 1 = 3


@pytest.mark.parametrize("tp_world_size", [1, 2])
@pytest.mark.parametrize("num_spec", [0, 1, 3, 7])
def test_short_conv_state_widens_by_num_spec(tp_world_size: int, num_spec: int):
    conv_dim = INTERMEDIATE_SIZE // tp_world_size
    (shape,) = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=tp_world_size,
        intermediate_size=INTERMEDIATE_SIZE,
        conv_kernel=CONV_KERNEL,
        num_spec=num_spec,
    )
    # Layout-agnostic: one axis is conv_dim, the other is the widened state_len.
    assert set(shape) == {conv_dim, CONV_KERNEL - 1 + num_spec}


def test_num_spec_defaults_to_zero():
    kwargs = dict(
        tp_world_size=1,
        intermediate_size=INTERMEDIATE_SIZE,
        conv_kernel=CONV_KERNEL,
    )
    assert MambaStateShapeCalculator.short_conv_state_shape(
        **kwargs
    ) == MambaStateShapeCalculator.short_conv_state_shape(**kwargs, num_spec=0)
