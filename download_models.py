"""Download the five sweep models into the local Hugging Face cache and check each one loads.

Nothing is timed or measured here. Run this once with network access; measure.py and the
sweep then run in offline mode and load only from the cache.
"""
import os

# --- Online mode: this script is the one place allowed to contact the Hub ---
# Set explicitly (before transformers is imported) in case offline mode is set in the shell.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import torch
from transformers import AutoModelForCausalLM

# Full Hub names, so the cache entries match exactly what the sweep will load.
MODELS = [
    "distilbert/distilgpt2",          # 82M,  train
    "openai-community/gpt2",          # 124M, train
    "openai-community/gpt2-medium",   # 355M, held out
    "Qwen/Qwen2.5-0.5B",              # 0.5B, train
    "TinyLlama/TinyLlama_v1.1",       # 1.1B, train (GQA + SwiGLU)
]

failed = []
for name in MODELS:
    print(f"\n=== {name} ===")
    try:
        # Downloads on first call, then loads from the cache. Same dtype and device as the sweep,
        # so a model that loads here will load there.
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")
        model.eval()

        cfg = model.config
        n_params = sum(p.numel() for p in model.parameters())
        # These are the model-shape features the predictor will use.
        hidden = getattr(cfg, "hidden_size", None) or cfg.n_embd
        layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
        heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
        kv_heads = getattr(cfg, "num_key_value_heads", None) or heads   # no GQA: one KV head per head
        print(f"  loaded OK: {n_params / 1e6:.0f}M params, hidden {hidden}, layers {layers}, "
              f"heads {heads}, KV heads {kv_heads}, max positions {cfg.max_position_embeddings}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        failed.append(name)
    finally:
        # Free GPU memory before the next model.
        model = None
        torch.cuda.empty_cache()

print()
if failed:
    raise SystemExit(f"{len(failed)} model(s) failed: {', '.join(failed)}")
print(f"All {len(MODELS)} models downloaded and load.")
