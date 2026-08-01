#!/bin/bash
#SBATCH --job-name=wf_lfm2diff_parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=gpuworker
#SBATCH --time=00:35:00
#SBATCH --mem=96G
#SBATCH --output=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/parity_%j.out
#SBATCH --error=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/parity_%j.out

SRC=/home/jens.luecke@liquid.ai/code/zwischenraum_vllm
CKPT=/home/jens.luecke@liquid.ai/code/wayfair-query-expansion/data/processed_v5c/diffusion/lfm2diff_sft_1627334
CKPT_HF=/home/shared_tw/pretrained/wqe_diffusion/sft_trl_1627334
OUTDIR=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/parity_out
WT_SRC=/home/jens.luecke@liquid.ai/code/wayfair-query-expansion/.claude/worktrees/diffusion/src
TMP=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp
CONTAINER_IMAGE="${CONTAINER_IMAGE:-vllm/vllm-openai-rocm:latest}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${HOME}:${HOME},/home/shared_tw:/home/shared_tw" \
     bash -lc '
set -e
VP=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)
echo "[overlay] container vllm at $VP"
cd '"$SRC"'
for f in $(git diff --name-only v0.26.0 HEAD -- "vllm/*"); do
  cp "$f" "$VP/${f#vllm/}"
done
echo "[overlay] applied $(git diff --name-only v0.26.0 HEAD -- "vllm/*" | wc -l) files"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CKPT='"$CKPT"' CKPT_HF='"$CKPT_HF"' OUTDIR='"$OUTDIR"' CANVAS=128

echo "===== HF REFERENCE ====="
PYTHONPATH='"$WT_SRC"' python3 '"$TMP"'/parity_hf.py

echo "===== VLLM ====="
export LFM2_DIFF_DUMP_LOGITS='"$OUTDIR"'/vllm_logits.pt
python3 '"$TMP"'/parity_vllm.py

echo "===== COMPARE ====="
python3 '"$TMP"'/parity_compare.py
echo "===== DONE_PARITY ====="
'
