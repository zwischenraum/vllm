"""vLLM side of the parity harness: dump first-step logits + generate."""

import json
import os

from vllm import LLM, SamplingParams

ckpt = os.environ["CKPT"]
outdir = os.environ["OUTDIR"]
prompts = json.load(open(os.path.join(outdir, "prompts.json")))

llm = LLM(
    model=ckpt,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    max_num_seqs=8,
    mamba_cache_mode="align",
)
sp = SamplingParams(max_tokens=256)

# Prompt 0 alone (batch of 1) so the one-shot logit dump captures its first
# denoising step over [prefix, all-MASK canvas].
llm.generate([{"prompt_token_ids": prompts[0]}], sp)

# All K prompts for the decoded-output comparison.
outs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sp)
res = [{"out": o.outputs[0].text} for o in outs]
json.dump(res, open(os.path.join(outdir, "vllm_outputs.json"), "w"), indent=2)
for i, o in enumerate(outs):
    print(f"[vllm] {i} out={o.outputs[0].text!r}", flush=True)
print("[vllm] DONE", flush=True)
