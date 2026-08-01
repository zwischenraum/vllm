"""Compare HF vs vLLM first-step logits + decoded outputs."""

import json
import os

import torch

outdir = os.environ["OUTDIR"]
REAL_VOCAB = 64402

hf = torch.load(os.path.join(outdir, "hf_logits.pt"))
vl = torch.load(os.environ["LFM2_DIFF_DUMP_LOGITS"])

hf_l = hf["logits"][0, :, :REAL_VOCAB].float()  # [CL, real_vocab]
vl_l = vl["logits"][0, :, :REAL_VOCAB].float()
canvas_all_mask = bool((vl["canvas"][0] == 64402).all().item())

print("=== BACKBONE LOGIT PARITY (first denoising step, [prefix, all-MASK]) ===")
print("vLLM first-step canvas all-[MASK]:", canvas_all_mask)
print("shapes hf/vllm:", tuple(hf_l.shape), tuple(vl_l.shape))
if hf_l.shape == vl_l.shape:
    diff = (hf_l - vl_l).abs()
    hf_am, vl_am = hf_l.argmax(-1), vl_l.argmax(-1)
    cos = torch.nn.functional.cosine_similarity(hf_l, vl_l, dim=-1)
    print(f"max|Δ|       = {diff.max().item():.4f}")
    print(f"mean|Δ|      = {diff.mean().item():.5f}")
    print(f"argmax agree = {(hf_am == vl_am).float().mean().item():.4f}")
    print(
        f"mean cosine  = {cos.mean().item():.5f}  min cosine = {cos.min().item():.5f}"
    )
    # per-position argmax agreement histogram over the canvas
    agree = hf_am == vl_am
    print(f"positions with matching argmax: {int(agree.sum())}/{agree.numel()}")

print("\n=== DECODED OUTPUT COMPARISON (diffusion_generate vs vLLM) ===")
hf_out = json.load(open(os.path.join(outdir, "hf_outputs.json")))
vl_out = json.load(open(os.path.join(outdir, "vllm_outputs.json")))
for i, h in enumerate(hf_out):
    v = vl_out[i]["out"] if i < len(vl_out) else "<none>"
    print(f"\n[{i}] q={h['query']!r}")
    print(f"    HF   (nfe={h['nfe']}): {h['out']!r}")
    print(f"    vLLM            : {v!r}")
print("\n[compare] DONE")
