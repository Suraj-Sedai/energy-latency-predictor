import os

# --- Offline mode: never contact the Hugging Face Hub, load only from the local cache ---
# Must be set before transformers is imported, because it reads these at import time.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time

import pynvml
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "gpt2"    # GPT-2 small, 124M params
BATCH_SIZE = 16
PROMPT_LEN = 128       # tokens per prompt
NEW_TOKENS = 128       # tokens generated per sequence: 1 from the prefill + 127 from decode
DECODE_STEPS = NEW_TOKENS - 1   # forced: no end-of-sequence check, so it always runs every step
SETTLE_SECONDS = 10.0  # fixed wait before measuring idle, so clocks/power drop back after model load
IDLE_SECONDS = 5.0     # how long to sit idle when measuring idle power
IDLE_SAMPLE_S = 0.1    # how often to sample power while idle
WARMUP_SECONDS = 2.0   # warm-up after idle: long enough for GPU clocks to ramp back up
IDLE_MAX_WATTS = 15.0  # idle above this means something else is using the GPU: don't trust the session
IDLE_ATTEMPTS = 3      # re-measure idle this many times before giving up
REPEATS = 10           # prefill+decode runs inside one energy window (one run is too short for NVML)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_state():
    """SM clock (MHz) and GPU temperature (C) right now. Both are cheap NVML reads."""
    return (pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))


def measure_idle():
    """Settle, then average power over IDLE_SECONDS. Returns (watts, sample count)."""
    # Fixed settle wait: loading the model, or a previous run, leaves the GPU in a high-power
    # state for a few seconds. Measuring straight away would read that tail as "idle".
    time.sleep(SETTLE_SECONDS)
    samples = []
    idle_start = time.perf_counter()
    while time.perf_counter() - idle_start < IDLE_SECONDS:
        samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)   # milliwatts -> watts
        time.sleep(IDLE_SAMPLE_S)
    return sum(samples) / len(samples), len(samples)


# --- Load the model in fp16 onto the GPU ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda")
model.eval()           # inference mode: turns off dropout

# --- Idle power: measured once per session, before any timed work ---
# Sync first so nothing queued (like the weight copy above) is still running.
# Idle check: on this laptop the GPU also drives the desktop, so a browser or other app can
# add load. A clean idle reads about 12 W. If it reads higher, wait and measure again; if it
# never comes down, stop rather than log a session with a wrong idle baseline.
torch.cuda.synchronize()
for attempt in range(1, IDLE_ATTEMPTS + 1):
    idle_watts, n_samples = measure_idle()
    clock, temp = gpu_state()
    print(f"Idle power: {idle_watts:.2f} W (mean of {n_samples} samples over {IDLE_SECONDS:.0f} s), "
          f"SM {clock} MHz, {temp} C  [attempt {attempt}/{IDLE_ATTEMPTS}]")
    if idle_watts <= IDLE_MAX_WATTS:
        break
else:
    pynvml.nvmlShutdown()
    raise SystemExit(f"Idle stayed above {IDLE_MAX_WATTS} W. Close apps using the GPU "
                     f"(browser, video, overlays) and run again.")

# --- Random token IDs: exact length, no tokenizer needed ---
# The GPU does the same work whatever the token values are, so random IDs are fine for timing.
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, PROMPT_LEN), device="cuda")


def decode(out, steps):
    """Greedy decode loop: one token per step, feeding the KV cache back in each time."""
    past = out.past_key_values                           # K and V for every prompt token, from the prefill
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (batch, 1): token 1, from the prefill
    for _ in range(steps):
        # Only the ONE new token goes in. The cache holds everything before it,
        # so the model never re-reads the prompt: it attends to the cached K/V instead.
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values                       # cache is now one token longer
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past


with torch.no_grad():  # no gradients: saves memory and time, we never train here
    # --- Warm-up, AFTER the idle measurement and BEFORE the timed runs ---
    # Sitting idle lets the GPU drop to low clocks. Without this, the first repeats would run
    # slow while the clocks climb back up. Full prefill+decode runs for WARMUP_SECONDS also pay
    # the one-time costs (CUDA kernel loading, memory allocation) for both prefill and decode kernels.
    warmup_start = time.perf_counter()
    warmup_runs = 0
    while time.perf_counter() - warmup_start < WARMUP_SECONDS:
        decode(model(input_ids, use_cache=True, logits_to_keep=1), steps=DECODE_STEPS)
        torch.cuda.synchronize()      # so the time check sees GPU work done, not just launched
        warmup_runs += 1
    print(f"Warm-up: {warmup_runs} runs over {time.perf_counter() - warmup_start:.1f} s")

    # --- Energy counter: read once before ALL repeats ---
    torch.cuda.synchronize()          # warm-up must be finished, or its energy leaks into ours
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)   # millijoules since driver load
    run_start = time.perf_counter()   # run duration spans exactly the two energy readings
    clock_start, temp_start = gpu_state()   # GPU state as the window opens

    # Repeat the whole run back to back inside one energy window. One run is ~0.6 s, close to
    # how often NVML updates its counter, so a single reading pair is noisy. Over REPEATS runs
    # that error is spread over a much longer window. Timings are taken per repeat and averaged.
    ttft_s, decode_s, states = [], [], []
    for _ in range(REPEATS):
        # --- Timed prefill: one forward pass over the whole prompt = TTFT ---
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Fresh prefill = fresh KV cache for this repeat. logits_to_keep=1: only the last position's
        # logits are computed, the only ones decode uses. The full (batch, prompt_len, vocab) logits
        # are wasted work and, at large batch x prompt length, gigabytes of memory.
        out = model(input_ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()      # wait for the GPU to actually finish before stopping the clock
        ttft_s.append(time.perf_counter() - start)

        # --- Timed decode: 127 steps, reusing this repeat's KV cache ---
        torch.cuda.synchronize()
        dec_start = time.perf_counter()
        past = decode(out, steps=DECODE_STEPS)
        torch.cuda.synchronize()      # each step is queued asynchronously; wait for all of them
        decode_s.append(time.perf_counter() - dec_start)
        states.append(gpu_state())    # SM clock and temperature at the end of this repeat

    # --- Energy counter: read once after ALL repeats ---
    torch.cuda.synchronize()          # no GPU work may still be running when we read the counter
    energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    run_end = time.perf_counter()
    clock_end, temp_end = gpu_state()       # GPU state as the window closes
    # Why the driver is holding clocks down right now (power cap, thermal, ...), as a bit mask.
    clock_reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)

pynvml.nvmlShutdown()

ttft_ms = sum(ttft_s) / REPEATS * 1000                 # mean over repeats
decode_ms = sum(decode_s) / REPEATS * 1000
tpot_ms = decode_ms / DECODE_STEPS            # time per decoded token
run_s = run_end - run_start                   # whole window, all repeats
tokens = REPEATS * BATCH_SIZE * NEW_TOKENS    # every token produced inside the window

# Total: everything the GPU drew during the run, idle draw included.
energy_j = (energy_after - energy_before) / 1000
energy_per_token_j = energy_j / tokens
# Above idle: subtract what the GPU would have burned anyway sitting idle for the same time.
energy_above_idle_j = energy_j - idle_watts * run_s
energy_above_idle_per_token_j = energy_above_idle_j / tokens

for i, ((ttft, dec), (clock, temp)) in enumerate(zip(zip(ttft_s, decode_s), states), 1):
    print(f"  repeat {i:2d}: TTFT {ttft * 1000:6.2f} ms, TPOT {dec / DECODE_STEPS * 1000:.3f} ms, "
          f"SM {clock} MHz, {temp} C")
print(f"Window: SM {clock_start} -> {clock_end} MHz, {temp_start} -> {temp_end} C, "
      f"clock event reasons 0x{clock_reasons:x}")
print(f"TTFT:   {ttft_ms:.2f} ms (mean of {REPEATS}, min {min(ttft_s) * 1000:.2f}, max {max(ttft_s) * 1000:.2f})")
print(f"Decode: {decode_ms:.2f} ms for {DECODE_STEPS} steps (mean of {REPEATS})")
print(f"TPOT:   {tpot_ms:.3f} ms")
print(f"Run:    {run_s * 1000:.2f} ms between energy readings, {tokens} tokens ({REPEATS} x {BATCH_SIZE} x {NEW_TOKENS})")
print(f"Energy (total):      {energy_j:.3f} J, {energy_per_token_j * 1000:.3f} mJ per token")
print(f"Energy (above idle): {energy_above_idle_j:.3f} J, {energy_above_idle_per_token_j * 1000:.3f} mJ per token")
# Sanity check (last repeat): the prompt plus the 127 tokens fed back in. The last predicted token is never fed in,
# so it isn't in the cache.
print("KV cache length:", past.get_seq_length(), f"(expected {PROMPT_LEN + DECODE_STEPS})")
