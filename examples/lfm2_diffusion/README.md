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
