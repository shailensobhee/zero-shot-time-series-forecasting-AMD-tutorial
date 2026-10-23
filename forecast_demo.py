#!/usr/bin/env python3
"""
Zero-shot time-series forecasting with Mistral AI's Timestral on an AMD GPU.

Loads Timestral-209M, forecasts a held-out window of a synthetic
electricity-load-style series with 21 probabilistic quantiles, scores the
forecast against ground truth, saves a plot, then benchmarks fleet-scale
(batched) forecasting throughput on a single AMD Instinct GPU.

Verified on an AMD Instinct MI300X (gfx942), torch 2.12.0+rocm7.14.0.

Usage:
    python forecast_demo.py                 # auto-selects cuda (ROCm) if available
    DEVICE=cpu python forecast_demo.py      # force CPU (Timestral runs CPU-only too)
"""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from timestral import TimestralForecaster

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MODEL = os.environ.get("MODEL", "mistralai/Timestral-209M")
OUT = os.environ.get("OUT", "forecast.png")


def to_np(x):
    """Timestral returns torch tensors on the model's device; bring them to host."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def main():
    dev_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"torch {torch.__version__} | HIP {torch.version.hip} | device={DEVICE} ({dev_name})")

    t0 = time.time()
    fc = TimestralForecaster.from_pretrained(MODEL, device=DEVICE)
    print(f"[load] {MODEL} -> {DEVICE} in {time.time() - t0:.1f}s")

    # --- build a realistic series: 15-minute electricity-style load ---------
    rng = np.random.default_rng(7)
    spd = 96                      # 15-min steps per day
    H = 21 * spd                  # 3 weeks of history as context
    horizon = 2 * spd             # forecast the next 2 days
    tt = np.arange(H + horizon)
    base = (
        50
        + 10 * np.sin(2 * np.pi * tt / spd - 1.2)        # daily cycle
        + 4 * np.sin(2 * np.pi * tt / (7 * spd))         # weekly cycle
        + 0.004 * tt                                     # slow upward trend
    )
    full = base + rng.normal(0, 1.5, size=tt.shape)
    context, truth = full[:H], full[H:H + horizon]
    ctx = context.astype(np.float32)[None, None, :]       # (series, variates, time)

    # --- forecast (warm the kernels once, then time the real call) ----------
    _ = fc.forecast(ctx, prediction_length=horizon)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = fc.forecast(ctx, prediction_length=horizon)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print(f"[forecast] 1 series, {horizon} steps in {(time.time() - t0) * 1000:.0f} ms")
    print(f"[shape] quantiles {tuple(out.quantiles.shape)}  (series, variates, horizon, 21 quantiles)")

    med = to_np(out.median)[0, 0]
    q10, q90 = to_np(out.quantile(0.1))[0, 0], to_np(out.quantile(0.9))[0, 0]
    q05, q95 = to_np(out.quantile(0.05))[0, 0], to_np(out.quantile(0.95))[0, 0]

    mae = float(np.mean(np.abs(med - truth)))
    naive = float(np.mean(np.abs(context[-1] - truth)))   # "repeat last value" baseline
    print(f"[accuracy] median MAE={mae:.2f} | naive-last MAE={naive:.2f} | skill={1 - mae / naive:.1%}")

    # --- plot ---------------------------------------------------------------
    plt.figure(figsize=(13, 5))
    hx = np.arange(-3 * spd, 0)
    plt.plot(hx, context[hx], color="#1f77b4", lw=1.2, label="history (last 3 days)")
    fx = np.arange(0, horizon)
    plt.plot(fx, truth, color="#333", lw=1.5, label="actual (held out)")
    plt.plot(fx, med, color="#d62728", lw=1.7, label="Timestral median")
    plt.fill_between(fx, q05, q95, color="#d62728", alpha=0.12, label="5-95% quantile")
    plt.fill_between(fx, q10, q90, color="#d62728", alpha=0.22, label="10-90% quantile")
    plt.axvline(0, color="k", ls=":", lw=0.8)
    plt.title(
        f"Timestral-209M zero-shot forecast on {dev_name}  |  "
        f"MAE {mae:.2f} vs naive {naive:.2f}  (skill {1 - mae / naive:.0%})"
    )
    plt.xlabel("15-min steps (0 = forecast start)")
    plt.ylabel("load")
    plt.legend(ncol=5, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT, dpi=130)
    print(f"[plot] saved {OUT}")

    # --- fleet-scale throughput benchmark -----------------------------------
    print("\n[benchmark] batched forecasting throughput")
    best = 0.0
    for B in [1, 64, 256, 1024]:
        batch = (base[None, None, :H] + rng.normal(0, 1.5, size=(B, 1, H))).astype(np.float32)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = fc.forecast(batch, prediction_length=horizon)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        best = max(best, B / dt)
        print(f"  {B:5d} series x {horizon} steps : {dt:6.2f} s = {B / dt:9.1f} series/s")
    print(f"\n[peak] {best:.0f} series/s on {dev_name}")
    print("DONE")


if __name__ == "__main__":
    main()
