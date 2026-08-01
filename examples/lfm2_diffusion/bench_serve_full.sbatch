#!/bin/bash
#SBATCH --job-name=wf_lfm2diff_benchfull
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=gpuworker
#SBATCH --time=00:55:00
#SBATCH --mem=96G
#SBATCH --output=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/benchfull_%j.out
#SBATCH --error=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/benchfull_%j.out

SRC=/home/jens.luecke@liquid.ai/code/zwischenraum_vllm
CKPT=/home/jens.luecke@liquid.ai/code/wayfair-query-expansion/data/processed_v5c/diffusion/lfm2diff_sft_1627334
QUERIES=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/wqe_bench_queries.jsonl
OUTDIR=/home/jens.luecke@liquid.ai/.claude/jobs/d4f91f8c/tmp/bench_out
CONTAINER_IMAGE="${CONTAINER_IMAGE:-vllm/vllm-openai-rocm:latest}"
PORT=$((26000 + SLURM_JOB_ID % 20000))
mkdir -p "$OUTDIR"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${HOME}:${HOME},/home/shared_tw:/home/shared_tw" \
     bash -lc '
set -e
VP=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)
cd '"$SRC"'
for f in $(git diff --name-only v0.26.0 HEAD -- "vllm/*"); do cp "$f" "$VP/${f#vllm/}"; done
echo "[overlay] applied to $VP"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING
PORT='"$PORT"'
OUTDIR='"$OUTDIR"'

CKPT='"$CKPT"'
echo "[serve] starting vllm serve on port $PORT"
vllm serve $CKPT --port $PORT \
    --max-model-len 512 --mamba-cache-mode align \
    --gpu-memory-utilization 0.6 > $OUTDIR/server.log 2>&1 &
SPID=$!

up=0
for i in $(seq 1 200); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[serve] SERVER_UP after ${i}0s"; up=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "[serve] SERVER_DIED"; tail -40 $OUTDIR/server.log; exit 1; fi
  sleep 10
done
if [ "$up" != "1" ]; then echo "[serve] SERVER_TIMEOUT"; tail -40 $OUTDIR/server.log; kill $SPID; exit 1; fi

for CONC in 1 8; do
  echo "===== BENCH conc=$CONC ====="
  vllm bench serve --backend vllm --host localhost --port $PORT --endpoint /v1/completions \
    --model $CKPT --tokenizer $CKPT --dataset-name custom --dataset-path '"$QUERIES"' \
    --num-prompts 128 --max-concurrency $CONC \
    --save-result --result-dir $OUTDIR --result-filename bench_full_conc${CONC}.json \
    --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tail -40
done

echo "[bench] BENCH_DONE"
kill $SPID 2>/dev/null || true
echo "===== DONE_BENCHSERVE ====="
'
