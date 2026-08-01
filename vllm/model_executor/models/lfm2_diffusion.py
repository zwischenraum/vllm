# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM2 masked-diffusion (block-diffusion) decoder for vLLM.

Adapts the stock LFM2 backbone into a masked absorbing-state (MDLM) diffusion
decoder, riding the generic diffusion-decode core that ships for DiffusionGemma
(``ModelConfig.is_diffusion`` -> V2 GPU model runner, per-sequence-causal
attention, zero-commit scheduler, ``ModelState`` hook).

YOCO two-phase structure (same backbone weights):
- encoder phase (``num_draft_tokens == 0``): the prompt prefix runs causally and
  writes KV + ShortConv state. Corresponds to the research harness's
  "prefix causal among itself, prefix blind to canvas".
- decoder phase (``num_draft_tokens == canvas_length``): the fixed-length canvas
  runs bidirectionally, reading the cached prefix KV/conv state and denoising in
  parallel. Corresponds to "canvas bidirectional over prefix+canvas".

Unlike DiffusionGemma (pure attention) LFM2 is hybrid, so the decoder phase runs
a multi-token ShortConv over the canvas that must read the cached prefix conv
state without advancing it (see ``short_conv.py`` diffusion path).

The sampler ports the Liquid ``diffusion_generate`` frozen-commit decoder:
top-2 margin confidence, threshold commit, frozen (never re-decoded) commits,
EOS/PAD suppression until ``min_content`` content tokens, annealed Gumbel
selection noise, temperature sampling. Self-conditioning is intentionally
dropped (the Liquid model defers it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, async_copy_to_gpu
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = init_logger(__name__)

# Config vocab is 65536; the real tokenizer uses 64402 ids, so row 64402 is the
# absorbing [MASK]. Logits for rows >= REAL_VOCAB_SIZE are forbidden.
REAL_VOCAB_SIZE = 64402
MASK_TOKEN_ID = 64402
EOS_ID = 7
PAD_ID = 0


# ---------------------------------------------------------------------------
# Model class: the stock LFM2 backbone + a ModelState hook.
# ---------------------------------------------------------------------------


class Lfm2DiffusionForCausalLM(Lfm2ForCausalLM):
    """LFM2 diffusion decoder.

    Reuses ``Lfm2ForCausalLM`` verbatim (backbone, tied lm_head, hybrid
    ShortConv/mamba state allocation via IsHybrid, standard weight loading -- the
    checkpoint is a plain LFM2 checkpoint whose row-64402 [MASK] embedding is
    already trained/mean-initialized). The only override routes vLLM onto the
    diffusion decode path via the ModelState hook.
    """

    @staticmethod
    def get_model_state_cls():
        return Lfm2DiffusionModelState


# ---------------------------------------------------------------------------
# Per-request GPU state (frozen-commit variant of DiffusionGemmaRequestStates).
# ---------------------------------------------------------------------------


class Lfm2DiffusionRequestStates:
    """Pre-allocated per-slot GPU tensors for the frozen-commit decoder."""

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        max_denoising_steps: int,
        mask_token_id: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.mask_token_id = mask_token_id
        self.device = device

        # True = encoder (prompt/commit) phase; False = denoise phase.
        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas token ids. Uncommitted positions hold [MASK].
        self.canvas = torch.full(
            (max_num_reqs, canvas_length),
            mask_token_id,
            dtype=torch.int64,
            device=device,
        )
        # Frozen-commit mask: a committed position is never re-decoded.
        self.committed = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.bool, device=device
        )
        # Denoising step counter (0..max_denoising_steps).
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Per-slot prompt length (set by add_request).
        self.prompt_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Reset the given slots' canvas to all-[MASK] and clear commits."""
        idx = torch.from_numpy(slot_indices_np.astype(np.int64)).to(self.device)
        self.canvas[idx] = self.mask_token_id
        self.committed[idx] = False
        self.step[idx] = 0

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.canvas[slot_idx] = self.mask_token_id
        self.committed[slot_idx] = False
        self.step[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False
        self.committed[slot_idx] = False


# ---------------------------------------------------------------------------
# Compiled per-step frozen-commit denoiser.
# ---------------------------------------------------------------------------


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True)
def _compiled_sample_step(
    logits: torch.Tensor,  # [num_decode * CL, vocab]
    decode_slots: torch.Tensor,  # [num_decode] int64
    decode_idx: torch.Tensor,  # [num_decode] int64
    all_slots: torch.Tensor,  # [num_reqs] int64
    valid_canvas_len: torch.Tensor,  # [num_decode] int64
    # State (mutated in place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    committed: torch.Tensor,  # [max_num_reqs, CL] bool
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs] bool
    # Output (mutated in place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: float,
    temperature: float,
    threshold: float,
    min_content: int,
    metric_is_margin: bool,
    real_vocab_size: int,
    eos_id: int,
    pad_id: int,
    CL: int,
) -> None:
    """One frozen-commit denoising step, vectorized over decode requests.

    Denoise step: commit all uncommitted positions whose confidence >= threshold
    (progress floor commits the single best if none qualify), freezing them; emit
    nothing (num_sampled=0). On convergence (all committed, or step budget hit)
    flip the slot to the commit phase. Commit step: emit the whole canvas.
    """
    num_decode = decode_slots.shape[0]
    device = decode_slots.device
    neg = float("-inf")

    logits_3d = logits.reshape(num_decode, CL, -1).float()
    logits_3d[..., real_vocab_size:] = neg  # forbid [MASK]/unused rows

    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit
    cur_canvas = canvas[decode_slots]  # [num_decode, CL]
    cur_committed = committed[decode_slots]  # [num_decode, CL]
    cur_step = step_tensor[decode_slots].float()  # [num_decode]

    # EOS/PAD suppression until `min_content` committed content tokens.
    is_content = cur_committed & (cur_canvas != eos_id) & (cur_canvas != pad_id)
    suppress = is_content.sum(dim=1) < min_content  # [num_decode]
    supp = suppress.view(num_decode, 1)
    logits_3d[..., eos_id] = torch.where(
        supp, torch.full_like(logits_3d[..., eos_id], neg), logits_3d[..., eos_id]
    )
    logits_3d[..., pad_id] = torch.where(
        supp, torch.full_like(logits_3d[..., pad_id], neg), logits_3d[..., pad_id]
    )

    probs = torch.softmax(logits_3d / max(temperature, 1e-6), dim=-1)
    if metric_is_margin:
        top2 = probs.topk(2, dim=-1).values
        conf = top2[..., 0] - top2[..., 1]  # [num_decode, CL]
    else:
        conf = probs.max(dim=-1).values
    conf = conf.masked_fill(cur_committed, neg)  # freeze committed positions

    # Annealed Gumbel selection noise (1 -> 0 over the step budget).
    sel_t = (1.0 - cur_step / max(max_denoising_steps - 1.0, 1.0)).clamp(min=0.0)
    u = torch.rand(conf.shape, device=device).clamp_(1e-9, 1.0)
    conf = conf + sel_t.view(num_decode, 1) * (-torch.log(-torch.log(u)))

    sampled_tok = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(
        num_decode, CL
    )

    newly = (conf >= threshold) & (~cur_committed)
    # Progress floor: a denoising row that would commit nothing commits its best.
    none_new = newly.sum(dim=1) == 0
    has_left = (~cur_committed).any(dim=1)
    stuck = none_new & has_left & is_denoise
    best = conf.argmax(dim=1)
    stuck_scatter = torch.zeros_like(newly)
    stuck_scatter[torch.arange(num_decode, device=device), best] = stuck
    newly = newly | stuck_scatter

    new_canvas = torch.where(newly, sampled_tok, cur_canvas)
    new_committed = cur_committed | newly
    new_step = torch.where(is_denoise, (cur_step + 1.0), torch.zeros_like(cur_step)).to(
        step_tensor.dtype
    )

    all_committed = new_committed.all(dim=1)
    converged = all_committed | (new_step.float() >= max_denoising_steps)

    # Write state: only denoise steps mutate canvas/committed/step.
    den = is_denoise.view(num_decode, 1)
    canvas[decode_slots] = torch.where(den, new_canvas, cur_canvas)
    committed[decode_slots] = torch.where(den, new_committed, cur_committed)
    step_tensor[decode_slots] = torch.where(
        is_denoise, new_step, step_tensor[decode_slots]
    )
    # Denoise: flip to commit-phase iff converged. Commit: leave False (done).
    is_encoder_phase[decode_slots] = torch.where(
        is_denoise, converged, torch.zeros_like(is_commit)
    )

    # Emit: commit steps emit the whole (fully committed) canvas.
    emit = is_commit.view(num_decode, 1)
    sampled[decode_idx] = torch.where(
        emit, cur_canvas, torch.zeros_like(cur_canvas)
    ).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    # Copy canvas -> draft_tokens for the next forward.
    draft_tokens[all_slots, :CL] = canvas[all_slots]


# ---------------------------------------------------------------------------
# ModelState.
# ---------------------------------------------------------------------------


class Lfm2DiffusionModelState(MambaHybridModelState):
    """ModelState for the LFM2 diffusion decoder.

    encoder mode (num_draft_tokens == 0): causal attention, writes KV/conv state.
    decoder mode (num_draft_tokens == canvas_length): bidirectional, reads state.
    """

    num_new_sampled_tokens_per_step: int = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 128
        text_config = self.model_config.hf_text_config
        self.gen_config = self.model_config.try_get_generation_config()
        max_denoising_steps = (
            diffusion_config.max_denoising_steps if diffusion_config else None
        ) or self.gen_config.get("max_denoising_steps", 32)
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.mask_token_id = getattr(text_config, "mask_token_id", MASK_TOKEN_ID)

        self.diffusion_states = Lfm2DiffusionRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            max_denoising_steps=max_denoising_steps,
            mask_token_id=self.mask_token_id,
            device=device,
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent per-request causal flags (in-place for CUDA graph replay).
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        gen = self.gen_config or {}
        return Lfm2DiffusionSampler(
            sampler=sampler,
            canvas_length=self.canvas_length,
            max_denoising_steps=self.max_denoising_steps,
            diffusion_states=self.diffusion_states,
            temperature=float(
                gen.get("diffusion_temperature", gen.get("temperature", 0.35))
            ),
            threshold=float(gen.get("commit_threshold", 0.4)),
            min_content=int(gen.get("min_content", 16)),
            metric=str(gen.get("confidence_metric", "margin")),
            real_vocab_size=int(
                getattr(
                    self.model_config.hf_text_config, "real_vocab_size", REAL_VOCAB_SIZE
                )
            ),
            eos_id=int(
                getattr(self.model_config.hf_text_config, "eos_token_id", EOS_ID)
                or EOS_ID
            ),
            pad_id=int(
                getattr(self.model_config.hf_text_config, "pad_token_id", PAD_ID)
                or PAD_ID
            ),
        ), None

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        # Mamba/ShortConv state bookkeeping (num_accepted, align seed).
        super().add_request(req_index, new_req_data)
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        if not new_req_data.req_id.startswith("_warmup_"):
            self.diffusion_states.prompt_len[req_index] = len(
                new_req_data.prompt_token_ids
            )

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ) -> dict[str, Any]:
        # Mirror MambaHybridModelState.prepare_attn (which supplies is_prefilling,
        # seq_lens_cpu_upper_bound, and the mamba/ShortConv spec metadata the conv
        # metadata builder requires) and additionally thread a per-request causal
        # tensor for the encoder/decoder attention phase split.
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

        # Per-request causal mode: encoder (prompt/commit) = causal,
        # denoise = bidirectional. Pass a GPU tensor for mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            causal=causal,
        )


# ---------------------------------------------------------------------------
# Sampler.
# ---------------------------------------------------------------------------


class Lfm2DiffusionSampler:
    """Frozen-commit parallel-unmasking sampler (port of diffusion_generate).

    Mirrors ``DiffusionSampler``'s prefill/decode plumbing; the per-step math is
    the frozen-commit ``_compiled_sample_step`` above (no self-conditioning, no
    entropy-bound remask).
    """

    def __init__(
        self,
        sampler: Any,
        canvas_length: int,
        max_denoising_steps: int,
        diffusion_states: Lfm2DiffusionRequestStates,
        *,
        temperature: float,
        threshold: float,
        min_content: int,
        metric: str,
        real_vocab_size: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.diffusion_states = diffusion_states
        self.temperature = temperature
        self.threshold = threshold
        self.min_content = min_content
        self.metric_is_margin = metric == "margin"
        self.real_vocab_size = real_vocab_size
        self.eos_id = eos_id
        self.pad_id = pad_id

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int32, device=device
        )
        self._num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        from types import SimpleNamespace

        return SimpleNamespace(output_bin_counts=None)

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        states.init_canvas(ps)
        self.req_states.draft_tokens[ps, : self.canvas_length] = states.canvas[ps]
        ps_gpu = async_copy_to_gpu(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(self, input_batch: Any, device: torch.device) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    def _build_output(
        self, input_batch, sampled, num_sampled, per_req_nlogits_np, device
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()
        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def __call__(
        self, logits: torch.Tensor, input_batch: Any, draft_logits=None
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()
        all_slots = input_batch.idx_mapping[:num_reqs]

        if num_decode > 0:
            _compiled_sample_step(
                logits[: num_decode * CL],
                decode_slots,
                decode_idx,
                all_slots,
                valid_canvas_len,
                states.canvas,
                states.committed,
                states.step,
                states.is_encoder_phase,
                sampled,
                num_sampled,
                self.req_states.draft_tokens,
                max_denoising_steps=float(self.max_denoising_steps),
                temperature=self.temperature,
                threshold=self.threshold,
                min_content=self.min_content,
                metric_is_margin=self.metric_is_margin,
                real_vocab_size=self.real_vocab_size,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
                CL=CL,
            )

        return self._build_output(
            input_batch, sampled, num_sampled, per_req_nlogits_np, device
        )
