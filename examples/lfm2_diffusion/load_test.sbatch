#!/bin/bash
#SBATCH --job-name=wf_lfm2diff_load
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=gpuworker
#SBATCH --time=00:25:00
#SBATCH --mem=96G
#SBATCH --output=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/load_%j.out
#SBATCH --error=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/load_%j.out

SRC=/home/jens.luecke@liquid.ai/code/zwischenraum_vllm
CKPT=/home/jens.luecke@liquid.ai/code/wayfair-query-expansion/data/processed_v5c/diffusion/lfm2diff_ckpt_s43
CONTAINER_IMAGE="${CONTAINER_IMAGE:-vllm/vllm-openai-rocm:latest}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${HOME}:${HOME},/home/shared_tw:/home/shared_tw" \
     bash -lc '
set -e
VP=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)
echo "[overlay] container vllm at $VP"
cd '"$SRC"'
for f in $(git diff --name-only v0.26.0 HEAD -- "vllm/*"); do
  rel=${f#vllm/}
  echo "[overlay] $f -> $VP/$rel"
  cp "$f" "$VP/$rel"
done
echo "[overlay] done"
export CKPT='"$CKPT"'
export VLLM_USE_V1=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python3 /home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/load_test.py
echo "=== DONE_LOAD_JOB ==="
'
