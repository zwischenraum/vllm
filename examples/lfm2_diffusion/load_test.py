"""Milestone A/B: construct the LFM2-diffusion engine and run one generate."""

import os
import traceback

from vllm import LLM, SamplingParams

ckpt = os.environ["CKPT"]
print(f"[load_test] constructing LLM for {ckpt}", flush=True)
try:
    llm = LLM(
        model=ckpt,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        max_num_seqs=4,
        mamba_cache_mode="align",
    )
    print("[load_test] ENGINE_CONSTRUCTED", flush=True)
except Exception:
    traceback.print_exc()
    print("[load_test] ENGINE_CONSTRUCT_FAILED", flush=True)
    raise

try:
    # Diffusion models reject per-request temperature/seed (read from
    # generation_config instead), so use default SamplingParams.
    out = llm.generate(
        ["blue velvet sofa", "kids bunk bed with storage"],
        SamplingParams(max_tokens=256),
    )
    for o in out:
        print("[load_test] OUT:", repr(o.outputs[0].text), flush=True)
    print("[load_test] GENERATE_OK", flush=True)
except Exception:
    traceback.print_exc()
    print("[load_test] GENERATE_FAILED", flush=True)

print("DONE_LOAD", flush=True)
