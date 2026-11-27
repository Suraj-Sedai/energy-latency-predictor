"""Energy/latency sweep: every model x batch size x prompt length x repeat, one CSV row per run.

    python sweep.py                                  # full sweep, all five models (~270 runs)
    python sweep.py --models openai-community/gpt2 --out data/dry_run.csv    # dry run
    python sweep.py --models TinyLlama/TinyLlama_v1.1 --batches 64 128 --out data/dry_run_big.csv

Models run one at a time (load, run all its configs, free). Within a model the configs run in a
shuffled order, so slow drift (heat, background load) spreads across configs instead of landing
on the last ones. Each row is written and flushed the moment its run finishes, so a crash loses
at most the run in progress. Out-of-memory is logged as a row, not a crash.
"""
import os

# --- Offline mode: load only from the local cache (download_models.py fills it) ---
# Must be set before transformers is imported, because it reads these at import time.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import random
import re
import time
from datetime import datetime

import pynvml
import torch
from transformers import AutoModelForCausalLM

MODELS = [
    "distilbert/distilgpt2",          # 82M,  train
    "openai-community/gpt2",          # 124M, train
    "openai-community/gpt2-medium",   # 355M, held out
    "Qwen/Qwen2.5-0.5B",              # 0.5B, train
    "TinyLlama/TinyLlama_v1.1",       # 1.1B, train (GQA + SwiGLU)
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]   # 64/128 reach the energy knee and OOM
PROMPT_LENS = [128, 256, 512]
REPEATS = 3            # rows per configuration

DTYPE = torch.float16
NEW_TOKENS = 128       # tokens generated per sequence: 1 from the prefill + 127 from decode
DECODE_STEPS = NEW_TOKENS - 1   # forced: no end-of-sequence check, so it always runs every step

SETTLE_SECONDS = 10.0  # fixed wait before measuring idle, so power drops back after model load
IDLE_SECONDS = 5.0     # how long to sit idle when measuring idle power
IDLE_SAMPLE_S = 0.1    # how often to sample power while idle
IDLE_MAX_WATTS = 12.5  # clean idle on this laptop reads ~11.6-12.4 W; above this something else is on the GPU
IDLE_ATTEMPTS = 3      # re-measure idle this many times before giving up

MODEL_WARMUP_S = 2.0   # discarded warm-up per model: kernels loaded, clocks back up after idle/load
MIN_WINDOW_S = 3.0     # each energy window repeats prefill+decode until it lasts at least this long.
                       # One run of a small model is ~0.6 s, too short for NVML's energy counter.

COLUMNS = [
    # --- run identity ---
    "timestamp", "gpu_name", "model", "repeat", "status", "error",
    # --- features ---
    "params", "hidden_size", "layers", "heads", "kv_heads", "dtype", "batch_size", "prompt_len",
    # --- targets ---
    "energy_per_token_j", "energy_above_idle_per_token_j", "ttft_s", "tpot_s",
    # --- sanity columns, not features ---
    "new_tokens", "inner_runs", "window_s", "energy_j", "energy_above_idle_j", "idle_watts",
    "ttft_min_s", "tpot_min_s", "peak_mem_gb", "wall_s",
    "temp_start_c", "temp_end_c", "sm_clock_start_mhz", "sm_clock_end_mhz", "clock_event_reasons",
]

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
GPU_NAME = torch.cuda.get_device_name(0)


def gpu_state():
    """SM clock (MHz) and GPU temperature (C) right now. Both are cheap NVML reads."""
    return (pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))


def energy_mj():
    """NVML total-energy counter, millijoules since driver load. Syncs first so no work is pending."""
    torch.cuda.synchronize()
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


def measure_idle():
    """Settle, then average power over IDLE_SECONDS. Returns (watts, sample count)."""
    torch.cuda.synchronize()
    time.sleep(SETTLE_SECONDS)
    samples = []
    start = time.perf_counter()
    while time.perf_counter() - start < IDLE_SECONDS:
        samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)   # milliwatts -> watts
        time.sleep(IDLE_SAMPLE_S)
    return sum(samples) / len(samples), len(samples)


def checked_idle():
    """Idle power once per session. On this laptop the GPU also drives the desktop, so another app
    can add load. If idle reads high, measure again; if it never comes down, stop the session."""
    for attempt in range(1, IDLE_ATTEMPTS + 1):
        watts, n = measure_idle()
        clock, temp = gpu_state()
        print(f"Idle power: {watts:.2f} W (mean of {n} samples over {IDLE_SECONDS:.0f} s), "
              f"SM {clock} MHz, {temp} C  [attempt {attempt}/{IDLE_ATTEMPTS}]")
        if watts <= IDLE_MAX_WATTS:
            return watts
    raise SystemExit(f"Idle stayed above {IDLE_MAX_WATTS} W. Close apps using the GPU "
                     f"(browser, video, overlays) and run again.")


def model_features(model):
    """The model-shape features the predictor uses. GPT-2 and Llama-style configs name them differently."""
    cfg = model.config
    heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "hidden_size": getattr(cfg, "hidden_size", None) or cfg.n_embd,
        "layers": getattr(cfg, "num_hidden_layers", None) or cfg.n_layer,
        "heads": heads,
        "kv_heads": getattr(cfg, "num_key_value_heads", None) or heads,   # no GQA: one KV head per head
        "dtype": str(DTYPE).replace("torch.", ""),
    }


def prefill(model, input_ids):
    # logits_to_keep=1: only the last position's logits, the only ones decode uses. The full
    # (batch, prompt_len, vocab) logits would be wasted work and gigabytes at batch 32 x 512.
    return model(input_ids, use_cache=True, logits_to_keep=1)


def decode(model, out, steps):
    """Greedy decode loop: one token per step, feeding the KV cache back in each time."""
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (batch, 1): token 1, from the prefill
    for _ in range(steps):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past


def run_config(model, batch_size, prompt_len, idle_watts):
    """One row: an energy window of back-to-back prefill+decode runs for one configuration."""
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, prompt_len), device="cuda")

    # Untimed warm-up for THIS shape: a new batch x length can pick different kernels.
    decode(model, prefill(model, input_ids), steps=8)

    torch.cuda.reset_peak_memory_stats()
    clock_start, temp_start = gpu_state()
    e_before = energy_mj()
    window_start = time.perf_counter()

    ttft_s, decode_s = [], []
    while not ttft_s or time.perf_counter() - window_start < MIN_WINDOW_S:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = prefill(model, input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        past = decode(model, out, steps=DECODE_STEPS)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ttft_s.append(t1 - t0)
        decode_s.append(t2 - t1)
        assert past.get_seq_length() == prompt_len + DECODE_STEPS    # every step reused the cache
        del out, past

    e_after = energy_mj()
    window_s = time.perf_counter() - window_start
    clock_end, temp_end = gpu_state()
    reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)

    runs = len(ttft_s)
    tokens = runs * batch_size * NEW_TOKENS
    energy_j = (e_after - e_before) / 1000
    above_idle_j = energy_j - idle_watts * window_s
    return {
        "energy_per_token_j": energy_j / tokens,
        "energy_above_idle_per_token_j": above_idle_j / tokens,
        "ttft_s": sum(ttft_s) / runs,
        "tpot_s": sum(decode_s) / runs / DECODE_STEPS,
        "inner_runs": runs,
        "window_s": window_s,
        "energy_j": energy_j,
        "energy_above_idle_j": above_idle_j,
        "ttft_min_s": min(ttft_s),
        "tpot_min_s": min(decode_s) / DECODE_STEPS,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "temp_start_c": temp_start, "temp_end_c": temp_end,
        "sm_clock_start_mhz": clock_start, "sm_clock_end_mhz": clock_end,
        "clock_event_reasons": f"0x{reasons:x}",
    }


def oom_dominated(batch_size, prompt_len, ooms):
    """True if a smaller-or-equal config already ran out of memory, so this one must too."""
    return any(batch_size >= b and prompt_len >= p for b, p in ooms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--batches", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=PROMPT_LENS)
    slug = re.sub(r"[^a-z0-9]+", "_", GPU_NAME.lower()).strip("_")
    parser.add_argument("--out", default=f"data/sweep_v2_{slug}.csv", help="one CSV per GPU")
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffled run order")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Never add rows to an existing sweep: a new run gets a new file, so no two sessions
    # (different idle baselines, different code) end up mixed in one CSV.
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        raise SystemExit(f"{args.out} already exists. Pick a new --out; old sweeps are never overwritten.")
    new_file = True
    f = open(args.out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    if new_file:
        writer.writeheader()

    def write_row(row):
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())          # on disk now, not sitting in a buffer

    print(f"GPU: {GPU_NAME}   ->  {args.out}")
    idle_watts = checked_idle()
    rng = random.Random(args.seed)
    total = len(args.models) * len(args.batches) * len(args.prompt_lens) * REPEATS
    done = 0
    sweep_start = time.perf_counter()

    for name in args.models:
        print(f"\n=== {name} ===")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to("cuda")
        model.eval()
        feats = model_features(model)

        configs = [(b, p, r) for b in args.batches for p in args.prompt_lens for r in range(1, REPEATS + 1)]
        rng.shuffle(configs)
        ooms = []                     # (batch, prompt_len) pairs that ran out of memory

        with torch.no_grad():
            # Discarded warm-up per model: kernels loaded, clocks back up after the load.
            t0 = time.perf_counter()
            ids = torch.randint(0, model.config.vocab_size, (4, 128), device="cuda")
            while time.perf_counter() - t0 < MODEL_WARMUP_S:
                decode(model, prefill(model, ids), steps=DECODE_STEPS)
                torch.cuda.synchronize()
            del ids

            for b, p, r in configs:
                row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "gpu_name": GPU_NAME,
                       "model": name, "repeat": r, "batch_size": b, "prompt_len": p,
                       "new_tokens": NEW_TOKENS, "idle_watts": idle_watts, "error": "", **feats}
                wall_start = time.perf_counter()
                if oom_dominated(b, p, ooms):
                    # Stop at out-of-memory: don't run what must fail, but keep it as a data point.
                    row["status"] = "oom_skipped"
                else:
                    try:
                        row.update(run_config(model, b, p, idle_watts))
                        row["status"] = "ok"
                    except torch.cuda.OutOfMemoryError as e:
                        row["status"] = "oom"
                        row["error"] = str(e).splitlines()[0][:200]
                        ooms.append((b, p))
                    torch.cuda.empty_cache()     # release what the run (or the failed run) held
                row["wall_s"] = time.perf_counter() - wall_start
                write_row(row)

                done += 1
                if row["status"] == "ok":
                    print(f"[{done:3d}/{total}] b={b:2d} p={p} r={r}  TTFT {row['ttft_s'] * 1000:7.2f} ms  "
                          f"TPOT {row['tpot_s'] * 1000:6.3f} ms  {row['energy_per_token_j'] * 1000:7.3f} mJ/tok  "
                          f"x{row['inner_runs']}  {row['temp_end_c']} C  {row['sm_clock_end_mhz']} MHz")
                else:
                    print(f"[{done:3d}/{total}] b={b:2d} p={p} r={r}  {row['status'].upper()}")

        del model
        torch.cuda.empty_cache()

    f.close()
    pynvml.nvmlShutdown()
    print(f"\nDone: {done} rows in {(time.perf_counter() - sweep_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
