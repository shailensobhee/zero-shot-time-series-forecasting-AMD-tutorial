#!/usr/bin/env python3
"""
Single-config Timestral CPU benchmark worker.

Thread count and NUMA binding are set BEFORE import via env/numactl by the
driver, so this worker only reads torch.get_num_threads() to report what it
actually got. It runs a fixed forecasting workload and prints one JSON line.

Config comes from env:
  TS_MODEL   e.g. mistralai/Timestral-209M
  TS_BATCH   number of independent series to forecast at once (int)
  TS_HORIZON forecast horizon in steps (default 192 = 2 days of 15-min)
  TS_HIST    history/context length in steps (default 2016 = 3 weeks)
  TS_REPEAT  timed repetitions (default 3), min taken
"""
import os, json, time, sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

MODEL = os.environ.get("TS_MODEL", "mistralai/Timestral-209M")
BATCH = int(os.environ.get("TS_BATCH", "1"))
HORIZON = int(os.environ.get("TS_HORIZON", "192"))
HIST = int(os.environ.get("TS_HIST", "2016"))
REPEAT = int(os.environ.get("TS_REPEAT", "3"))

from timestral import TimestralForecaster

torch_threads = torch.get_num_threads()

t0 = time.time()
fc = TimestralForecaster.from_pretrained(MODEL, device="cpu")
load_s = time.time() - t0

rng = np.random.default_rng(0)
spd = 96
tt = np.arange(HIST)
base = 50 + 10*np.sin(2*np.pi*tt/spd - 1.2) + 4*np.sin(2*np.pi*tt/(7*spd)) + 0.004*tt
batch = (base[None, None, :] + rng.normal(0, 1.5, size=(BATCH, 1, HIST))).astype(np.float32)

# warmup (JIT/oneDNN primitive selection)
with torch.inference_mode():
    _ = fc.forecast(batch, prediction_length=HORIZON)

times = []
with torch.inference_mode():
    for _ in range(REPEAT):
        t0 = time.time()
        out = fc.forecast(batch, prediction_length=HORIZON)
        times.append(time.time() - t0)

best = min(times)
rec = {
    "model": MODEL.split("/")[-1],
    "torch_threads": torch_threads,
    "env_omp": os.environ.get("OMP_NUM_THREADS"),
    "numa": os.environ.get("TS_NUMA_LABEL", "default"),
    "batch": BATCH,
    "horizon": HORIZON,
    "hist": HIST,
    "load_s": round(load_s, 2),
    "best_s": round(best, 4),
    "mean_s": round(sum(times)/len(times), 4),
    "series_per_s": round(BATCH / best, 1),
    "ms_per_series": round(best / BATCH * 1000, 2),
}
print("RESULT " + json.dumps(rec))
