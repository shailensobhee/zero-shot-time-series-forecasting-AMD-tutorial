# Timestral on AMD CPU: benchmark results and tuning

Timestral is designed to run on **CPU or GPU**. The original announcement for the model highlighted CPU inference (a one-day forecast in ~10 ms for the 5M model, ~140 ms for the 209M model on an 8-thread CPU). This document reproduces and tunes that CPU story on an **AMD EPYC** server, and puts a sample of AMD Instinct MI300X GPU numbers next to it for context.

Everything here was measured on real AMD hardware. Nothing is estimated.

## Hardware and software

**CPU (the focus of this document)**
- 2-socket **AMD EPYC** (Zen4-class engineering sample), **256 physical cores** total (128 per socket), SMT off
- 2 NUMA nodes (node0 = cores 0-127, node1 = cores 128-255), ~3 TB RAM
- AVX-512 with BF16 and VNNI, 1 GiB L3
- PyTorch **2.14.1+cpu** (oneDNN / MKL-DNN enabled), Python 3.11, Timestral standalone inference package

**GPU (for the side-by-side sample)**
- 1x **AMD Instinct MI300X** (gfx942), PyTorch **2.12.0 + ROCm 7.14**

**Workload (identical on CPU and GPU)**
- Context (history): 2016 steps (3 weeks of 15-minute data)
- Forecast horizon: 192 steps (2 days)
- Timed with a warmup pass, then the minimum of several repetitions

## Headline: the announcement's CPU claim reproduces on AMD EPYC

| Model | Reported (announcement, 8-thread CPU) | Measured on AMD EPYC (single-series latency) |
|---|---|---|
| Timestral-5M | ~10 ms | **9.3 ms** (8 threads, 1 NUMA node) |
| Timestral-209M | ~140 ms | **125 ms** (16 threads, 1 NUMA node) |

The small model answers a single forecast in **under 10 milliseconds** on AMD EPYC, with no GPU involved.

## Tuning findings (what actually matters on a 2-socket EPYC)

### 1. Bind to a single NUMA node. This is the biggest lever.
Spreading threads across both sockets with interleaved memory was consistently **slower** than staying on one node, because the per-forecast working set is small and cross-socket memory traffic dominates. Timestral-5M at batch 256:

| Placement | Threads | Throughput |
|---|---|---|
| **1 NUMA node** (`numactl --cpunodebind=0 --membind=0`) | 32 | **684 series/s** |
| Both sockets, interleaved (`numactl --interleave=all`) | 256 | 358 series/s |

Using all 256 cores across both sockets was about **half** the throughput of 32 cores pinned to one node. More cores is not more speed here.

### 2. There is a thread sweet spot, and it is modest.
Beyond roughly **8-32 threads per inference** the models stop scaling and then regress (thread sync and memory bandwidth outweigh added compute). Single-series latency, Timestral-209M:

| Threads | Latency |
|---|---|
| 1 | 667 ms |
| 4 | 261 ms |
| 8 | 141 ms |
| **16** | **125 ms** (best) |
| 32 | 128 ms |
| 64 | 137 ms |

Timestral-5M bottoms out even earlier, at **8 threads (9.3 ms)**.

### 3. For many series, run several small jobs in parallel, one per NUMA node.
Because a single forecast peaks at a modest thread count, the way to use all 256 cores is **not** one giant job. Pin a worker to each NUMA node (and optionally several workers per node, each with ~8-32 threads) and forecast different series in parallel. On this box that means up to 2 node-local workers at full memory bandwidth, or more smaller workers for latency-bound serving.

### 4. Smaller model, much higher throughput. Pick the smallest that meets your accuracy bar.
Peak throughput on one NUMA node (32 threads, batch 256):

| Model | Throughput | ms/series |
|---|---|---|
| Timestral-5M | **684 series/s** | 1.46 |
| Timestral-28M | 191 series/s | 5.23 |
| Timestral-123M | 30 series/s | 32.9 |
| Timestral-209M | 20 series/s | 49.3 |

The 5M model is roughly **34x** the throughput of the 209M model on CPU. All four sit on the accuracy/size Pareto front, so for high-volume CPU forecasting the small variants are very attractive.

## Recommended CPU recipe (AMD EPYC)

```bash
# Pin to ONE NUMA node, cap threads at the sweet spot, bind memory locally.
OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close \
numactl --cpunodebind=0 --membind=0 --physcpubind=0-15 \
  python your_forecast.py
```

- Start at **8 threads for the 5M model**, **16 for the 209M model**, then tune +/- for your series length.
- Keep each worker **inside one NUMA node**. Never let a single job straddle both sockets.
- To use the whole machine, launch **one worker per NUMA node** (here `--cpunodebind=0` and `--cpunodebind=1`) and split your series between them.
- PyTorch CPU wheels ship oneDNN, which already uses the EPYC AVX-512 / BF16 paths. No extra math library is required.

## Sample results: AMD EPYC CPU next to AMD Instinct MI300X GPU

Same model, same 3-week context, same 2-day horizon. CPU numbers are the best single-NUMA-node configuration; GPU numbers are one MI300X.

### Single-series latency (batch = 1)

| Model | AMD EPYC CPU (best) | AMD Instinct MI300X |
|---|---|---|
| Timestral-5M | 9.3 ms (8 threads) | 17.6 ms |
| Timestral-209M | 125 ms (16 threads) | 33.6 ms |

For a **single** small forecast the CPU is actually competitive (even faster for the 5M model), because the work is tiny and a GPU launch has fixed overhead. This is exactly why the model ships as CPU-friendly.

### Batched throughput (batch = 256 independent series)

| Model | AMD EPYC CPU (best) | AMD Instinct MI300X |
|---|---|---|
| Timestral-5M | 684 series/s | 10,939 series/s |
| Timestral-209M | 24 series/s | 1,072 series/s |

When you batch many series, the GPU's parallelism pulls far ahead (about **16x** for the 5M model and **45x** for the 209M model at batch 256, and the MI300X gap widens further at larger batches).

### How to read this

- **Low-volume, latency-sensitive, or edge / embedded** forecasting: AMD EPYC CPU is an excellent fit. A single forecast is milliseconds, no accelerator needed, and the 5M model is tiny (21 MB).
- **High-volume fleet forecasting** (thousands of series, frequent refresh): AMD Instinct MI300X, which forecasts the whole fleet in one batched pass.
- Same open model, same code, same AMD software stack across both. You choose the hardware to match the job.

## Reproduce it

The benchmark harness is in [`bench/`](bench/):

- `cpu_bench_worker.py` runs one `(model, threads, batch)` configuration and prints a JSON result. Thread count and NUMA binding are set by the launcher before PyTorch imports.
- `cpu_sweep.sh` runs the full sweep used for this document (latency vs threads, throughput vs threads, NUMA scaling, and a model-size sweep).

```bash
cd bench
bash cpu_sweep.sh results.jsonl
```
