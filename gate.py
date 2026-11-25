import pynvml
import torch
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# --- First: does PyTorch see the GPU? ---
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("PyTorch cannot see the GPU. Stop here, this must be fixed first.")
    pynvml.nvmlShutdown()
    exit()

# --- Idle reading ---
power_idle = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
print("Idle power (W):", power_idle)

# --- Matmul load: multiply big matrices for a few seconds ---
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")

start = time.time()
while time.time() - start < 5:        # hammer the GPU for 5 seconds
    c = a @ b                          # matrix multiply on the GPU
torch.cuda.synchronize()              # wait for all GPU work to actually finish

# --- Loaded reading ---
power_load = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

print("Load power (W):", power_load)
print("Energy used in 5s (J):", (energy_after - energy_before) / 1000)

pynvml.nvmlShutdown()