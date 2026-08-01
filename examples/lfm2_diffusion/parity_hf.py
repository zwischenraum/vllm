"""HF reference for the LFM2-diffusion vLLM parity harness.

Emits:
  - hf_logits.pt: first denoising-step logits over [prefix, all-MASK canvas] for
    prompt 0 (deterministic; isolates the backbone forward = attn mask + ShortConv).
  - hf_outputs.json: full diffusion_generate decoded outputs for K prompts.
  - prompts.json: the exact prefix token ids used (so vLLM uses identical inputs).
"""

import json
import os

import torch
from transformers import AutoTokenizer
from wayfair_query_expansion.diffusion.model import (
    MASK_TOKEN_ID,
    build_diffusion_attention_mask,
    load_diffusion_lfm2,
)
from wayfair_query_expansion.diffusion.sampler import decode_canvas, diffusion_generate

# HF loads the ORIGINAL LFM2 checkpoint (model_type=lfm2); the vLLM-only
# lfm2_diffusion config is not recognized by plain transformers. Same weights.
CKPT = os.environ["CKPT_HF"]
OUTDIR = os.environ["OUTDIR"]
CL = int(os.environ.get("CANVAS", "128"))
EOS_ID = 7

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(CKPT)
# init_mask=False: the [MASK] row is already trained in this checkpoint.
model = load_diffusion_lfm2(
    CKPT, dtype=torch.bfloat16, device=device, symmetric_conv=False, init_mask=False
)

# prompts.json (list of prefix_ids) is pre-extracted on the headnode so the
# container needs no pyarrow.
prompts = json.load(open(os.path.join(OUTDIR, "prompts.json")))

# --- prompt 0: first-step logits over [prefix, all-MASK canvas] ---
p0 = torch.tensor([prompts[0]], dtype=torch.long, device=device)
p = p0.shape[1]
canvas = torch.full((1, CL), MASK_TOKEN_ID, dtype=torch.long, device=device)
mask4d = build_diffusion_attention_mask(p, CL, dtype=torch.bfloat16, device=device)
pos = torch.arange(p + CL, device=device).unsqueeze(0)
with torch.inference_mode():
    out = model(
        input_ids=torch.cat([p0, canvas], dim=1),
        attention_mask=mask4d,
        position_ids=pos,
        use_cache=False,
    )
hf_logits = out.logits[:, p:, :].float().cpu()  # [1, CL, vocab]
torch.save({"logits": hf_logits, "prefix_len": p}, os.path.join(OUTDIR, "hf_logits.pt"))
print(
    f"[hf] saved first-step logits {tuple(hf_logits.shape)} prompt0 (prefix_len={p})"
)

# --- full diffusion_generate for K prompts ---
gen = torch.Generator(device=device).manual_seed(0)
outputs = []
for i, ids in enumerate(prompts):
    prefix = torch.tensor([ids], dtype=torch.long, device=device)
    query = tok.decode(ids[1:])  # drop BOS
    canvas_out, nfe = diffusion_generate(
        model,
        prefix,
        canvas_len=CL,
        max_steps=32,
        temperature=0.35,
        metric="margin",
        threshold=0.4,
        min_content=16,
        generator=gen,
    )
    text = decode_canvas(canvas_out[0], tok)
    outputs.append({"query": query, "nfe": int(nfe), "out": text})
    print(f"[hf] q={query!r} nfe={nfe} out={text!r}")
json.dump(outputs, open(os.path.join(OUTDIR, "hf_outputs.json"), "w"), indent=2)
print("[hf] DONE")
