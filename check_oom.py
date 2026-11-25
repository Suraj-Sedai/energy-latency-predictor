import torch

# Your GPU has 12 GB. Try to grab ~20 GB in one shot.
# Fallback OFF  -> this raises OutOfMemoryError (what we want).
# Fallback ON   -> this quietly succeeds by spilling into system RAM (bad).

total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU total memory: {total_gb:.1f} GB")

try:
    # float32 = 4 bytes; 5 billion elements ≈ 20 GB
    big = torch.empty(5_000_000_000, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    print("Allocation SUCCEEDED — fallback is likely ON. This must be turned off.")
    print("Allocated (GB):", big.element_size() * big.numel() / 1e9)
except torch.cuda.OutOfMemoryError:
    print("OutOfMemoryError — good. Fallback is OFF, OOM behaves correctly.")
