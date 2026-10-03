"""
Opening the Black Box: PyTorch Hooks into a Running LLM
========================================================
This script loads Llama 3.1 8B on an A100 GPU and attaches
forward/backward hooks to inspect activations, attention patterns,
gradient flow, and layer-by-layer behavior during inference.

Run on gpu002: ~/miniconda3/envs/llm/bin/python 01_llm_hooks_inference.py
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/models")
OUTPUT_DIR = os.path.expanduser("~/hook_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "Explain how a transformer model processes text step by step."

print("=" * 70)
print("OPENING THE BLACK BOX: PyTorch Hooks into Llama 3.1 8B")
print("=" * 70)

# ============================================================
# 1. Load Model and Tokenizer
# ============================================================
print("\n[1/7] Loading model and tokenizer...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"  Device: {next(model.parameters()).device}")
print(f"  Architecture: {model.config.architectures}")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Attention heads: {model.config.num_attention_heads}")

# ============================================================
# 2. Forward Hooks: Capture Activations
# ============================================================
print("\n[2/7] Attaching forward hooks to capture activations...")

activations = {}
activation_stats = {}

def make_activation_hook(name):
    def hook(module, input, output):
        # Handle tuple outputs (common in transformer layers)
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        if isinstance(tensor, torch.Tensor):
            activations[name] = tensor.detach().cpu().float()
            activation_stats[name] = {
                "shape": list(tensor.shape),
                "mean": tensor.float().mean().item(),
                "std": tensor.float().std().item(),
                "min": tensor.float().min().item(),
                "max": tensor.float().max().item(),
                "norm": tensor.float().norm().item(),
                "zeros_pct": (tensor == 0).float().mean().item() * 100,
            }
    return hook

# Attach to every decoder layer + embeddings + final norm
handles = []

# Embedding layer
h = model.model.embed_tokens.register_forward_hook(make_activation_hook("embed_tokens"))
handles.append(h)

# Each transformer layer
for i, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_activation_hook(f"layer_{i}"))
    handles.append(h)
    # Also hook into attention and MLP sub-modules
    h = layer.self_attn.register_forward_hook(make_activation_hook(f"layer_{i}_attn"))
    handles.append(h)
    h = layer.mlp.register_forward_hook(make_activation_hook(f"layer_{i}_mlp"))
    handles.append(h)

# Final layer norm
h = model.model.norm.register_forward_hook(make_activation_hook("final_norm"))
handles.append(h)

print(f"  Attached {len(handles)} forward hooks")

# ============================================================
# 3. Run Inference with Hooks Active
# ============================================================
print(f"\n[3/7] Running inference...")
print(f'  Prompt: "{PROMPT}"')

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]
print(f"  Input tokens: {input_ids.shape[1]}")

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
    )
gen_time = time.time() - t0

generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
new_tokens = outputs.sequences.shape[1] - input_ids.shape[1]
print(f"  Generated {new_tokens} tokens in {gen_time:.1f}s ({new_tokens/gen_time:.1f} tok/s)")
print(f"  Response: {generated_text[:200]}...")

# ============================================================
# 4. Analyze Activation Statistics
# ============================================================
print("\n[4/7] Analyzing activation statistics...")

print(f"\n  {'Layer':<25} {'Shape':<25} {'Mean':>8} {'Std':>8} {'Min':>10} {'Max':>10} {'Norm':>10} {'Dead%':>6}")
print("  " + "-" * 100)
for name, stats in sorted(activation_stats.items()):
    shape_str = str(stats["shape"])
    print(f"  {name:<25} {shape_str:<25} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
          f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['norm']:>10.2f} {stats['zeros_pct']:>5.1f}%")

# ============================================================
# 5. Visualize: Activation Norms Across Layers
# ============================================================
print("\n[5/7] Generating visualizations...")

# Plot 1: Activation norms per layer
layer_names = [f"layer_{i}" for i in range(model.config.num_hidden_layers)]
layer_norms = [activation_stats[n]["norm"] for n in layer_names if n in activation_stats]
layer_means = [activation_stats[n]["mean"] for n in layer_names if n in activation_stats]
layer_stds = [activation_stats[n]["std"] for n in layer_names if n in activation_stats]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Opening the Black Box: Llama 3.1 8B Layer Analysis", fontsize=16, fontweight="bold")

# Norms
ax = axes[0, 0]
ax.bar(range(len(layer_norms)), layer_norms, color="steelblue", alpha=0.8)
ax.set_xlabel("Layer Index")
ax.set_ylabel("Activation Norm")
ax.set_title("Activation Norms Across Layers")
ax.grid(axis="y", alpha=0.3)

# Means
ax = axes[0, 1]
ax.plot(range(len(layer_means)), layer_means, "o-", color="coral", markersize=3)
ax.set_xlabel("Layer Index")
ax.set_ylabel("Mean Activation")
ax.set_title("Mean Activation Value per Layer")
ax.grid(alpha=0.3)

# Std dev
ax = axes[1, 0]
ax.plot(range(len(layer_stds)), layer_stds, "s-", color="forestgreen", markersize=3)
ax.set_xlabel("Layer Index")
ax.set_ylabel("Activation Std Dev")
ax.set_title("Activation Standard Deviation per Layer")
ax.grid(alpha=0.3)

# Attention vs MLP norms
attn_norms = [activation_stats[f"layer_{i}_attn"]["norm"] for i in range(model.config.num_hidden_layers) if f"layer_{i}_attn" in activation_stats]
mlp_norms = [activation_stats[f"layer_{i}_mlp"]["norm"] for i in range(model.config.num_hidden_layers) if f"layer_{i}_mlp" in activation_stats]
x = range(min(len(attn_norms), len(mlp_norms)))
ax = axes[1, 1]
ax.plot(x, attn_norms[:len(x)], "o-", label="Attention", color="royalblue", markersize=3)
ax.plot(x, mlp_norms[:len(x)], "s-", label="MLP", color="tomato", markersize=3)
ax.set_xlabel("Layer Index")
ax.set_ylabel("Activation Norm")
ax.set_title("Attention vs MLP Activation Norms")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_activation_analysis.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {OUTPUT_DIR}/01_activation_analysis.png")

# ============================================================
# 6. Hook: Bottleneck Detection
# ============================================================
print("\n[6/7] Bottleneck detection...")

bottlenecks = []
for name, stats in activation_stats.items():
    if stats["zeros_pct"] > 50:
        bottlenecks.append((name, stats["zeros_pct"]))
    if stats["std"] < 1e-5:
        bottlenecks.append((name, f"near-zero std: {stats['std']:.2e}"))

if bottlenecks:
    print("  Potential bottlenecks found:")
    for name, issue in bottlenecks:
        print(f"    {name}: {issue}")
else:
    print("  No bottlenecks detected - all layers have healthy activations")

# ============================================================
# 7. Hook: Layer Timing (Performance Profiling)
# ============================================================
print("\n[7/7] Layer timing profiling...")

# Remove old hooks
for h in handles:
    h.remove()

# Attach timing hooks
layer_times = {}

def make_timing_hook(name):
    def hook(module, input, output):
        # We record end time; start time comes from pre-hook
        layer_times[name]["end"] = time.perf_counter()
    return hook

def make_pre_timing_hook(name):
    def hook(module, input):
        layer_times[name] = {"start": time.perf_counter(), "end": 0}
    return hook

timing_handles = []
for i, layer in enumerate(model.model.layers):
    name = f"layer_{i}"
    h1 = layer.register_forward_pre_hook(make_pre_timing_hook(name))
    h2 = layer.register_forward_hook(make_timing_hook(name))
    timing_handles.extend([h1, h2])

# Run a forward pass for timing
with torch.no_grad():
    _ = model(**inputs)

# Calculate durations
durations = {}
for name, times in layer_times.items():
    if times["end"] > 0:
        durations[name] = (times["end"] - times["start"]) * 1000  # ms

if durations:
    sorted_durations = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    total = sum(durations.values())

    print(f"\n  {'Layer':<20} {'Time (ms)':>10} {'% of Total':>12}")
    print("  " + "-" * 45)
    for name, dur in sorted_durations[:10]:
        print(f"  {name:<20} {dur:>10.3f} {dur/total*100:>11.1f}%")
    print(f"  {'TOTAL':<20} {total:>10.3f} {'100.0%':>12}")

    # Plot timing
    fig, ax = plt.subplots(figsize=(14, 5))
    layer_indices = [int(n.split("_")[1]) for n in sorted(durations.keys())]
    times_ms = [durations[f"layer_{i}"] for i in layer_indices]
    ax.bar(layer_indices, times_ms, color="mediumpurple", alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-Layer Inference Time: Llama 3.1 8B")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_layer_timing.png"), dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {OUTPUT_DIR}/02_layer_timing.png")

# Clean up timing hooks
for h in timing_handles:
    h.remove()

# ============================================================
# Save summary
# ============================================================
summary = {
    "model": MODEL_ID,
    "prompt": PROMPT,
    "generated_tokens": new_tokens,
    "generation_time_s": gen_time,
    "tokens_per_second": new_tokens / gen_time,
    "load_time_s": load_time,
    "total_parameters_B": sum(p.numel() for p in model.parameters()) / 1e9,
    "num_layers": model.config.num_hidden_layers,
    "hidden_size": model.config.hidden_size,
    "activation_stats": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in activation_stats.items()},
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved: {OUTPUT_DIR}/summary.json")

print("\n" + "=" * 70)
print("DONE. All hook outputs saved to:", OUTPUT_DIR)
print("=" * 70)
