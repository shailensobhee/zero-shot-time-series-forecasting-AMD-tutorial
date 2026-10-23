#!/usr/bin/env bash
# Timestral CPU sweep driver for a 2-socket AMD EPYC (volcano).
# Each config runs as a FRESH numactl-bound process, because thread count and
# NUMA placement must be fixed before torch imports. Results stream as JSON.
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/timestral-cpu/.venv/bin/activate

WORKER=cpu_bench_worker.py
OUT=${1:-/home/shailens/timestral-cpu/results.jsonl}
: > "$OUT"

run() {  # run <numa_label> <numactl-args> <threads> <model> <batch>
  local label="$1" na="$2" thr="$3" model="$4" batch="$5"
  OMP_NUM_THREADS=$thr OMP_PLACES=cores OMP_PROC_BIND=close \
  MKL_NUM_THREADS=$thr TS_NUMA_LABEL="$label" TS_MODEL="$model" \
  TS_BATCH=$batch TS_REPEAT=3 \
  numactl $na python "$WORKER" 2>/dev/null | grep '^RESULT ' | sed 's/^RESULT //' | tee -a "$OUT"
}

echo "### A. single-series LATENCY vs threads (batch=1)"
for model in mistralai/Timestral-5M mistralai/Timestral-209M; do
  for thr in 1 4 8 16 32 64; do
    run "node0" "--cpunodebind=0 --membind=0 --physcpubind=0-$((thr-1))" "$thr" "$model" 1
  done
done

echo "### B. THROUGHPUT vs threads, batch=256, single NUMA node"
for model in mistralai/Timestral-5M mistralai/Timestral-209M; do
  for thr in 8 16 32 64 128; do
    run "node0" "--cpunodebind=0 --membind=0 --physcpubind=0-$((thr-1))" "$thr" "$model" 256
  done
done

echo "### C. NUMA scaling: 256 threads across BOTH sockets, interleaved memory"
for model in mistralai/Timestral-5M mistralai/Timestral-209M; do
  for batch in 256 1024; do
    run "both-interleave" "--interleave=all --physcpubind=0-255" 256 "$model" $batch
  done
done

echo "### D. model-size sweep at the sweet spot (node0, 32 threads, batch=256)"
for model in mistralai/Timestral-5M mistralai/Timestral-28M mistralai/Timestral-123M mistralai/Timestral-209M; do
  run "node0" "--cpunodebind=0 --membind=0 --physcpubind=0-31" 32 "$model" 256
done

echo "### DONE -> $OUT"
