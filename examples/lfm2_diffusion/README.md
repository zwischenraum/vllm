# LFM2 diffusion — vLLM parity + smoke harness

Validates the `Lfm2DiffusionForBlockDiffusion` vLLM model against the reference
Liquid `diffusion_generate` (HF) harness, on the ROCm `vllm/vllm-openai-rocm`
(v0.26.0) container with this branch's files overlaid.

- `build_diff_ckpt.py` — turn an AR/warmup/SFT LFM2 checkpoint into a
  `lfm2_diffusion` checkpoint dir (patch config.json/generation_config.json;
  `--mean-init-mask` only when adapting from a checkpoint whose `[MASK]` row is
  untrained). Weights are symlinked.
- `parity_hf.py` / `parity_vllm.py` / `parity_compare.py` — the S6 backbone
  logit-parity harness (first denoising step over `[prefix, all-MASK]`, no
  sampling) + a decoded-output comparison. HF loads the original `lfm2`
  checkpoint (plain transformers can't load `model_type=lfm2_diffusion`); vLLM
  loads the patched dir. `parity_vllm.py` dumps first-step logits via
  `LFM2_DIFF_DUMP_LOGITS`.
- `parity.sbatch` / `load_test.sbatch` — SLURM drivers (attribution:
  `--wckey=v1/customer_engagement/wayfair/vlm_catalogue`).

## Result (SFT checkpoint `sft_trl_1627334`, MI325X)
- Backbone first-step logit parity vs HF: **argmax agree 126/128**, mean cosine
  0.92. Divergence is confined to **canvas positions 0–1** — the ShortConv
  prefix→canvas boundary is not yet seeded from the cached prefix conv state
  (the open S4 item).
- End-to-end decode reproduces the HF reference's v5c-format output per query.

## Fixes validated
- **Single-canvas termination**: emit the canvas up to the first EOS (force a
  trailing EOS when a full canvas has none) so the request stops after one canvas
  instead of re-blocking. Output no longer duplicates.
- **ShortConv prefix-state preservation (S4)**: the canvas denoise is a prefill
  with an initial state; causal_conv1d_fn wrote the advanced canvas state back,
  corrupting the prefix state for later steps. Snapshot+restore keeps the prefix
  state across denoise steps, which measurably improved decode coherence.
  (Residual: canvas positions 0-1 still diverge from HF at step 1 — a boundary
  item needing kernel-level conv-state debugging.)

## Serve benchmark (`bench_serve.sbatch`, SFT ckpt, MI325X, enforce-eager, 128 WQE queries)
| conc | req/s | out tok/s | median E2EL | P99 E2EL |
|-----:|------:|----------:|------------:|---------:|
| 1    | 10.9  | 1116      | 55.9 ms     | 229 ms   |
| 8    | 20.5  | 2031      | 71.4 ms     | 5163 ms  |

TPOT=0 / TTFT==E2EL is expected (the whole canvas commits at once — no token
streaming). Median 56 ms at conc=1 (eager) is near the <50 ms target a
cudagraph/compiled forward would clear.

## Compiled + PIECEWISE cudagraph (`bench_serve_cg.sbatch`, same ckpt/queries)
| conc | median E2EL (eager → compiled) | notes |
|-----:|-------------------------------:|-------|
| 1    | 55.9 → **41.3 ms** (<50 ms target) | steady-state; mean/P99 inflated by torch.compile warmup (first requests recompile) |
| 8    | 71.4 → 71.3 ms                 | piecewise + eager conv; throughput lower under compile warmup |

vLLM auto-selects FULL cudagraph, which the mamba backend rejects for diffusion:
`mamba_attn.py:183` asserts `max_query_len == 1 + num_spec_tokens` (AR spec-decode
= 129), but the diffusion canvas is exactly `num_spec_tokens` = 128 (no +1 bonus
token, since `num_new_sampled_tokens_per_step = 0`). PIECEWISE sidesteps it
(compile attention/MLP, conv eager). FULL cudagraph — capturing the whole hot
canvas forward — needs a diffusion-aware relaxation of that assertion.
