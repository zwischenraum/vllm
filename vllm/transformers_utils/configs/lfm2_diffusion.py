# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for the LFM2 masked-diffusion (block-diffusion) decoder.

The LFM2 diffusion decoder reuses the stock LFM2 backbone verbatim (full-attention
+ ShortConv layers, tied embeddings, vocab) and adapts it into a masked
absorbing-state (MDLM) diffusion decoder. The only structural additions live in
serving, not in the weights: a fixed-length denoising ``canvas`` and an iterative
confidence-based unmasking sampler.

``Lfm2DiffusionConfig`` therefore subclasses ``Lfm2Config`` and adds only the
fields vLLM's generic diffusion-decode path keys on. Exposing ``canvas_length`` is
what makes ``ModelConfig.is_diffusion`` return True, which routes the model onto
the zero-commit scheduler + per-sequence-causal attention rails shared with
DiffusionGemma.
"""

from typing import Any

from transformers import Lfm2Config


class Lfm2DiffusionConfig(Lfm2Config):
    model_type = "lfm2_diffusion"

    def __init__(
        self,
        canvas_length: int = 128,
        max_denoising_steps: int | None = 32,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        # Set the diffusion fields before delegating to Lfm2Config so they are
        # present as attributes regardless of how PretrainedConfig processes
        # kwargs (mirrors DiffusionGemmaConfig).
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.self_conditioning_size = self_conditioning_size
        super().__init__(**kwargs)
