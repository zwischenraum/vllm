"""Build a config-patched LFM2-diffusion checkpoint dir from an AR checkpoint.

For the load/forward smoke test we only need the config to route the checkpoint
onto the diffusion architecture; weights are symlinked as-is (the [MASK] mean-init
matters only for numerical parity, added later). Optionally mean-inits the
[MASK] row when --mean-init-mask is passed (writes a fresh safetensors).
"""

import argparse
import json
import os

REAL_VOCAB_SIZE = 64402
MASK_TOKEN_ID = 64402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--canvas-length", type=int, default=128)
    ap.add_argument("--max-denoising-steps", type=int, default=32)
    ap.add_argument("--mean-init-mask", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Symlink everything except the files we rewrite.
    rewrite = {"config.json", "generation_config.json"}
    if args.mean_init_mask:
        rewrite.add("model.safetensors")
    for name in os.listdir(args.src):
        if name in rewrite:
            continue
        src = os.path.join(args.src, name)
        dst = os.path.join(args.dst, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    # Patch config.json.
    with open(os.path.join(args.src, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["architectures"] = ["Lfm2DiffusionForBlockDiffusion"]
    cfg["model_type"] = "lfm2_diffusion"
    cfg["canvas_length"] = args.canvas_length
    cfg["max_denoising_steps"] = args.max_denoising_steps
    with open(os.path.join(args.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # Patch generation_config.json with diffusion sampler knobs.
    gen_path = os.path.join(args.src, "generation_config.json")
    gen = {}
    if os.path.exists(gen_path):
        with open(gen_path) as fh:
            gen = json.load(fh)
    gen.update(
        {
            "diffusion_temperature": 0.35,
            "commit_threshold": 0.4,
            "min_content": 16,
            "confidence_metric": "margin",
            "max_denoising_steps": args.max_denoising_steps,
        }
    )
    with open(os.path.join(args.dst, "generation_config.json"), "w") as fh:
        json.dump(gen, fh, indent=2)

    if args.mean_init_mask:
        import torch
        from safetensors.torch import load_file, save_file

        st = load_file(os.path.join(args.src, "model.safetensors"))
        emb_key = None
        for k in st:
            if k.endswith("embed_tokens.weight"):
                emb_key = k
                break
        if emb_key is None:
            raise SystemExit(f"embed_tokens.weight not found in {list(st)[:5]}")
        w = st[emb_key].to(torch.float32)
        w[MASK_TOKEN_ID] = w[:REAL_VOCAB_SIZE].mean(dim=0)
        st[emb_key] = w.to(st[emb_key].dtype)
        save_file(st, os.path.join(args.dst, "model.safetensors"))
        print(f"[mean-init] {emb_key} row {MASK_TOKEN_ID} set to mean of trained rows")

    print(f"[ok] wrote diffusion checkpoint dir: {args.dst}")
    print("  architectures:", cfg["architectures"], "model_type:", cfg["model_type"])
    print(
        "  canvas_length:",
        cfg["canvas_length"],
        "max_denoising_steps:",
        cfg["max_denoising_steps"],
    )


if __name__ == "__main__":
    main()
