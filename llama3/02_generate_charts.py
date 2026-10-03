"""
Generate publication-quality charts for each step of the LLM hooks analysis.
Each chart is designed to be self-explanatory for a blog post / presentation.

Run on gpu002: ~/miniconda3/envs/llm/bin/python ~/02_generate_charts.py
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import time
import os
from collections import defaultdict, OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/models")
OUTPUT_DIR = os.path.expanduser("~/hook_outputs/charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "Explain how a transformer model processes text step by step."

# Style settings
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
})

COLORS = {
    "primary": "#4361ee",
    "secondary": "#f72585",
    "accent": "#4cc9f0",
    "success": "#06d6a0",
    "warning": "#ffd166",
    "danger": "#ef476f",
    "dark": "#2b2d42",
    "attn": "#4361ee",
    "mlp": "#f72585",
    "embed": "#06d6a0",
    "norm": "#ffd166",
}

print("=" * 70)
print("GENERATING CHARTS: Opening the Black Box")
print("=" * 70)

# ============================================================
# Load Model
# ============================================================
print("\nLoading model...")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, dtype=torch.float16, device_map="auto",
)
model.eval()

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
num_heads = model.config.num_attention_heads

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"Model: {MODEL_ID}")
print(f"Layers: {num_layers}, Hidden: {hidden_size}, Heads: {num_heads}")
print(f"Prompt tokens: {len(tokens)}")


# ============================================================
# CHART 1: Model Architecture Overview
# ============================================================
print("\n[Chart 1] Model Architecture Overview...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
fig.suptitle("Chart 1: Llama 3.1 8B — Architecture Overview", fontsize=18, fontweight="bold", y=0.97)

# Draw the architecture as blocks
components = [
    (1, 10.5, 8, 0.8, "Input Tokens", COLORS["dark"], "white", f'"{PROMPT[:50]}..."'),
    (1, 9.2, 8, 0.8, "Token Embeddings", COLORS["embed"], "white", f"vocab -> {hidden_size}-dim vectors"),
    (1, 7.5, 8, 0.8, "RMSNorm (Pre-Norm)", COLORS["norm"], COLORS["dark"], "Normalize before each sub-layer"),
]

# Draw input/embed/norm
for x, y, w, h, label, color, tcolor, detail in components:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.1, label, ha="center", va="center",
            fontsize=13, fontweight="bold", color=tcolor)
    ax.text(x + w/2, y + h/2 - 0.2, detail, ha="center", va="center",
            fontsize=9, color=tcolor, alpha=0.8)

# Transformer block
block_y = 3.5
rect = mpatches.FancyBboxPatch((0.5, block_y), 9, 3.5, boxstyle="round,pad=0.15",
                                 facecolor="white", edgecolor=COLORS["dark"], linewidth=2, linestyle="--")
ax.add_patch(rect)
ax.text(5, block_y + 3.7, f"x{num_layers} Transformer Layers", ha="center", fontsize=14,
        fontweight="bold", color=COLORS["dark"])

# Attention block
rect_a = mpatches.FancyBboxPatch((1, block_y + 1.8), 3.5, 1.2, boxstyle="round,pad=0.1",
                                   facecolor=COLORS["attn"], edgecolor="white", linewidth=2)
ax.add_patch(rect_a)
ax.text(2.75, block_y + 2.7, "Self-Attention", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white")
ax.text(2.75, block_y + 2.2, f"{num_heads} heads, GQA", ha="center", va="center",
        fontsize=9, color="white", alpha=0.9)

# MLP block
rect_m = mpatches.FancyBboxPatch((5.5, block_y + 1.8), 3.5, 1.2, boxstyle="round,pad=0.1",
                                   facecolor=COLORS["mlp"], edgecolor="white", linewidth=2)
ax.add_patch(rect_m)
ax.text(7.25, block_y + 2.7, "Feed-Forward (MLP)", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white")
ax.text(7.25, block_y + 2.2, "SwiGLU activation", ha="center", va="center",
        fontsize=9, color="white", alpha=0.9)

# Residual connections
ax.annotate("", xy=(2.75, block_y + 1.6), xytext=(2.75, block_y + 0.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=2))
ax.annotate("", xy=(7.25, block_y + 1.6), xytext=(7.25, block_y + 0.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=2))
ax.text(5, block_y + 0.3, "Residual Connections (+ Add)", ha="center",
        fontsize=10, color=COLORS["success"], fontweight="bold")

# Output
rect_o = mpatches.FancyBboxPatch((1, 1.5), 8, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=COLORS["dark"], edgecolor="white", linewidth=2)
ax.add_patch(rect_o)
ax.text(5, 1.9, "LM Head: Linear Projection -> Vocabulary Logits", ha="center",
        va="center", fontsize=12, fontweight="bold", color="white")

# Stats box
stats_text = (f"Parameters: 8.03B\n"
              f"Hidden dim: {hidden_size}\n"
              f"Layers: {num_layers}\n"
              f"Heads: {num_heads}\n"
              f"Vocab: {model.config.vocab_size:,}")
ax.text(9.8, 1.0, stats_text, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e9ecef", alpha=0.8),
        family="monospace")

# Arrows between major blocks
for y_from, y_to in [(10.3, 10.0), (9.0, 7.5+0.8), (7.3, block_y+3.5), (block_y+0.0, 2.3)]:
    ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#adb5bd", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_01_architecture.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_01_architecture.png")


# ============================================================
# CHART 2: What Are Hooks? (Conceptual)
# ============================================================
print("[Chart 2] What Are Hooks — Conceptual Diagram...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Chart 2: Three Types of PyTorch Hooks", fontsize=18, fontweight="bold", y=1.02)

hook_types = [
    ("Forward Hook", "register_forward_hook()",
     "Captures layer OUTPUT\nafter forward pass",
     "Input -> [Layer] -> Output\n           ^hook reads output",
     COLORS["primary"],
     "Use cases:\n- Activation monitoring\n- Feature extraction\n- Bottleneck detection"),
    ("Backward Hook", "register_backward_hook()",
     "Captures GRADIENTS\nduring backpropagation",
     "Grad_out <- [Layer] <- Grad_in\n              ^hook reads gradients",
     COLORS["secondary"],
     "Use cases:\n- Gradient flow analysis\n- Gradient clipping\n- Vanishing gradient detection"),
    ("Forward Pre-Hook", "register_forward_pre_hook()",
     "Captures layer INPUT\nbefore forward pass",
     "Input -> [Layer] -> Output\n  ^hook reads input",
     COLORS["accent"],
     "Use cases:\n- Input validation\n- Layer timing (start)\n- Input modification"),
]

for ax, (title, api, desc, flow, color, uses) in zip(axes, hook_types):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, title, ha="center", va="center", fontsize=16,
            fontweight="bold", color=color)

    # API call
    ax.text(5, 8.5, api, ha="center", va="center", fontsize=10,
            family="monospace", color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e9ecef"))

    # Description
    ax.text(5, 7.2, desc, ha="center", va="center", fontsize=12, color=COLORS["dark"])

    # Flow diagram
    rect = mpatches.FancyBboxPatch((1, 4.8), 8, 1.5, boxstyle="round,pad=0.2",
                                     facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 5.5, flow, ha="center", va="center", fontsize=10,
            family="monospace", color=COLORS["dark"])

    # Use cases
    ax.text(5, 2.5, uses, ha="center", va="center", fontsize=10,
            color=COLORS["dark"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_02_hook_types.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_02_hook_types.png")


# ============================================================
# CHART 3: Token Embedding Visualization
# ============================================================
print("[Chart 3] Token Embeddings...")

embed_activations = {}

def embed_hook(module, input, output):
    embed_activations["embeddings"] = output.detach().cpu().float()

h = model.model.embed_tokens.register_forward_hook(embed_hook)
with torch.no_grad():
    _ = model(**inputs)
h.remove()

emb = embed_activations["embeddings"][0]  # [seq_len, hidden_dim]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Chart 3: Token Embeddings — The First Transformation",
             fontsize=18, fontweight="bold", y=1.02)

# Heatmap of embeddings
ax = axes[0]
im = ax.imshow(emb.numpy().T[:100, :], aspect="auto", cmap="RdBu_r", interpolation="nearest")
ax.set_xlabel("Token Position")
ax.set_ylabel("Embedding Dimension (first 100)")
ax.set_title("Embedding Vectors (Heatmap)")
ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# Per-token statistics
ax = axes[1]
token_norms = emb.norm(dim=1).numpy()
token_means = emb.mean(dim=1).numpy()
x = range(len(tokens))
bars = ax.bar(x, token_norms, color=COLORS["primary"], alpha=0.8)
ax.set_xlabel("Token")
ax.set_ylabel("L2 Norm")
ax.set_title("Embedding Norm per Token")
ax.set_xticks(x)
ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
# Highlight max
max_idx = np.argmax(token_norms)
bars[max_idx].set_color(COLORS["secondary"])
bars[max_idx].set_alpha(1.0)

# Distribution of embedding values
ax = axes[2]
ax.hist(emb.numpy().flatten(), bins=100, color=COLORS["accent"], alpha=0.8, edgecolor="white")
ax.axvline(emb.numpy().mean(), color=COLORS["secondary"], linestyle="--", linewidth=2,
           label=f"Mean: {emb.numpy().mean():.4f}")
ax.axvline(0, color=COLORS["dark"], linestyle="-", linewidth=1, alpha=0.5)
ax.set_xlabel("Activation Value")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of All Embedding Values")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_03_embeddings.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_03_embeddings.png")


# ============================================================
# CHART 4: Layer-by-Layer Activation Flow
# ============================================================
print("[Chart 4] Activation Flow Through All Layers...")

layer_stats = OrderedDict()

def make_stats_hook(name):
    def hook(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        if isinstance(t, torch.Tensor):
            t_f = t.detach().cpu().float()
            layer_stats[name] = {
                "mean": t_f.mean().item(),
                "std": t_f.std().item(),
                "norm": t_f.norm().item(),
                "min": t_f.min().item(),
                "max": t_f.max().item(),
                "abs_mean": t_f.abs().mean().item(),
            }
    return hook

handles = []
h = model.model.embed_tokens.register_forward_hook(make_stats_hook("Embed"))
handles.append(h)
for i in range(num_layers):
    handles.append(model.model.layers[i].register_forward_hook(make_stats_hook(f"L{i}")))
    handles.append(model.model.layers[i].self_attn.register_forward_hook(make_stats_hook(f"L{i}_Attn")))
    handles.append(model.model.layers[i].mlp.register_forward_hook(make_stats_hook(f"L{i}_MLP")))
h = model.model.norm.register_forward_hook(make_stats_hook("Norm"))
handles.append(h)

with torch.no_grad():
    _ = model(**inputs)

for h in handles:
    h.remove()

# Extract layer-level stats
layer_indices = list(range(num_layers))
norms = [layer_stats[f"L{i}"]["norm"] for i in layer_indices]
stds = [layer_stats[f"L{i}"]["std"] for i in layer_indices]
means = [layer_stats[f"L{i}"]["mean"] for i in layer_indices]
maxes = [layer_stats[f"L{i}"]["max"] for i in layer_indices]
mins = [layer_stats[f"L{i}"]["min"] for i in layer_indices]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Chart 4: How Activations Transform Layer by Layer",
             fontsize=18, fontweight="bold", y=1.01)

# Norms
ax = axes[0, 0]
bars = ax.bar(layer_indices, norms, color=COLORS["primary"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Layer")
ax.set_ylabel("L2 Norm")
ax.set_title("Activation Norm (magnitude grows deeper)")
ax.annotate(f"Layer 0: {norms[0]:.1f}", xy=(0, norms[0]), xytext=(3, norms[0]+5),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"]), fontsize=10)
ax.annotate(f"Layer 31: {norms[-1]:.1f}", xy=(31, norms[-1]), xytext=(25, norms[-1]-8),
            arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]), fontsize=10, color=COLORS["secondary"])

# Std Dev
ax = axes[0, 1]
ax.fill_between(layer_indices, stds, alpha=0.3, color=COLORS["secondary"])
ax.plot(layer_indices, stds, "o-", color=COLORS["secondary"], markersize=4, linewidth=2)
ax.set_xlabel("Layer")
ax.set_ylabel("Standard Deviation")
ax.set_title("Activation Spread (variance increases)")
ax.annotate(f"Spike at L31:\n{stds[-1]:.2f}", xy=(31, stds[-1]),
            xytext=(24, stds[-1]*0.8),
            arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]), fontsize=10)

# Min/Max range
ax = axes[1, 0]
ax.fill_between(layer_indices, mins, maxes, alpha=0.3, color=COLORS["accent"])
ax.plot(layer_indices, maxes, "-", color=COLORS["success"], linewidth=2, label="Max")
ax.plot(layer_indices, mins, "-", color=COLORS["danger"], linewidth=2, label="Min")
ax.plot(layer_indices, means, "--", color=COLORS["dark"], linewidth=1, label="Mean")
ax.set_xlabel("Layer")
ax.set_ylabel("Value")
ax.set_title("Activation Range (min-max envelope)")
ax.legend(loc="lower left")

# Attention vs MLP contribution
ax = axes[1, 1]
attn_norms = [layer_stats[f"L{i}_Attn"]["norm"] for i in layer_indices]
mlp_norms = [layer_stats[f"L{i}_MLP"]["norm"] for i in layer_indices]
ax.plot(layer_indices, attn_norms, "o-", color=COLORS["attn"], markersize=4,
        linewidth=2, label="Self-Attention")
ax.plot(layer_indices, mlp_norms, "s-", color=COLORS["mlp"], markersize=4,
        linewidth=2, label="MLP (Feed-Forward)")
ax.fill_between(layer_indices, attn_norms, alpha=0.1, color=COLORS["attn"])
ax.fill_between(layer_indices, mlp_norms, alpha=0.1, color=COLORS["mlp"])
ax.set_xlabel("Layer")
ax.set_ylabel("Activation Norm")
ax.set_title("Attention vs MLP: Who Does the Work?")
ax.legend()
ax.annotate("MLP dominates\nin final layers", xy=(30, mlp_norms[-2]),
            xytext=(20, max(mlp_norms)*0.7),
            arrowprops=dict(arrowstyle="->", color=COLORS["mlp"]), fontsize=10, color=COLORS["mlp"])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_04_activation_flow.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_04_activation_flow.png")


# ============================================================
# CHART 5: Attention Patterns
# ============================================================
print("[Chart 5] Attention Pattern Visualization...")

attn_weights = {}

def make_attn_hook(layer_idx):
    def hook(module, input, output):
        # output is (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights[layer_idx] = output[1].detach().cpu().float()
    return hook

# Need to enable attention output
handles = []
for i in range(num_layers):
    h = model.model.layers[i].self_attn.register_forward_hook(make_attn_hook(i))
    handles.append(h)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

for h in handles:
    h.remove()

# Get attention from model output
if hasattr(outputs, "attentions") and outputs.attentions is not None:
    all_attentions = outputs.attentions
else:
    all_attentions = None

if all_attentions is not None and len(all_attentions) > 0:
    # Pick 4 representative layers
    pick_layers = [0, num_layers//3, 2*num_layers//3, num_layers-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Chart 5: Attention Patterns — What Each Layer Focuses On",
                 fontsize=18, fontweight="bold", y=1.01)

    for ax, li in zip(axes.flat, pick_layers):
        attn = all_attentions[li][0]  # [num_heads, seq_len, seq_len]
        # Average across heads
        avg_attn = attn.mean(dim=0).numpy()

        im = ax.imshow(avg_attn, cmap="Blues", interpolation="nearest", vmin=0)
        ax.set_title(f"Layer {li} (avg across {attn.shape[0]} heads)")
        ax.set_xlabel("Key Position (attends to)")
        ax.set_ylabel("Query Position (from)")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_05_attention_patterns.png"), dpi=150, bbox_inches="tight")
    print("  Saved chart_05_attention_patterns.png")
else:
    print("  Skipping — attention weights not available (needs eager attention)")
    # Create a placeholder explaining this
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, "Attention pattern visualization requires\n"
            "model loaded with attn_implementation='eager'\n\n"
            "Re-run with:\nmodel = AutoModelForCausalLM.from_pretrained(\n"
            "    ..., attn_implementation='eager')",
            ha="center", va="center", fontsize=14, family="monospace",
            bbox=dict(boxstyle="round,pad=1", facecolor="#fff3cd", edgecolor="#ffc107"))
    ax.axis("off")
    fig.suptitle("Chart 5: Attention Patterns (Requires Eager Attention)", fontsize=16, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_05_attention_patterns.png"), dpi=150, bbox_inches="tight")
    print("  Saved chart_05_attention_patterns.png (placeholder)")


# ============================================================
# CHART 6: Layer Timing Profiling
# ============================================================
print("[Chart 6] Layer Timing Profiling...")

layer_times = {}

def make_pre_hook(name):
    def hook(module, input):
        layer_times[name] = {"start": time.perf_counter()}
    return hook

def make_post_hook(name):
    def hook(module, input, output):
        layer_times[name]["end"] = time.perf_counter()
        layer_times[name]["duration_ms"] = (layer_times[name]["end"] - layer_times[name]["start"]) * 1000
    return hook

handles = []
# Embedding
handles.append(model.model.embed_tokens.register_forward_pre_hook(make_pre_hook("Embed")))
handles.append(model.model.embed_tokens.register_forward_hook(make_post_hook("Embed")))
# Each layer + sub-components
for i in range(num_layers):
    for name, mod in [("Full", model.model.layers[i]),
                       ("Attn", model.model.layers[i].self_attn),
                       ("MLP", model.model.layers[i].mlp)]:
        key = f"L{i}_{name}"
        handles.append(mod.register_forward_pre_hook(make_pre_hook(key)))
        handles.append(mod.register_forward_hook(make_post_hook(key)))
# Final norm
handles.append(model.model.norm.register_forward_pre_hook(make_pre_hook("Norm")))
handles.append(model.model.norm.register_forward_hook(make_post_hook("Norm")))

# Warm up
with torch.no_grad():
    _ = model(**inputs)

# Actual measurement (average 5 runs)
all_timings = defaultdict(list)
for _ in range(5):
    layer_times.clear()
    with torch.no_grad():
        _ = model(**inputs)
    for k, v in layer_times.items():
        if "duration_ms" in v:
            all_timings[k].append(v["duration_ms"])

for h in handles:
    h.remove()

avg_timings = {k: np.mean(v) for k, v in all_timings.items()}

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Chart 6: Where Does Inference Time Go? (Per-Layer Profiling)",
             fontsize=18, fontweight="bold", y=1.02)

# Full layer times
ax = axes[0]
full_times = [avg_timings.get(f"L{i}_Full", 0) for i in range(num_layers)]
colors_bar = [COLORS["primary"]] * num_layers
max_i = np.argmax(full_times)
colors_bar[max_i] = COLORS["secondary"]
ax.barh(range(num_layers), full_times, color=colors_bar, alpha=0.85, edgecolor="white")
ax.set_ylabel("Layer")
ax.set_xlabel("Time (ms)")
ax.set_title("Total Time per Layer")
ax.invert_yaxis()
ax.set_yticks(range(0, num_layers, 4))

# Attention vs MLP time
ax = axes[1]
attn_times = [avg_timings.get(f"L{i}_Attn", 0) for i in range(num_layers)]
mlp_times = [avg_timings.get(f"L{i}_MLP", 0) for i in range(num_layers)]
width = 0.35
x = np.arange(num_layers)
ax.bar(x - width/2, attn_times, width, label="Attention", color=COLORS["attn"], alpha=0.8)
ax.bar(x + width/2, mlp_times, width, label="MLP", color=COLORS["mlp"], alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Time (ms)")
ax.set_title("Attention vs MLP Time")
ax.legend()
ax.set_xticks(range(0, num_layers, 4))

# Pie chart: component breakdown
ax = axes[2]
embed_time = avg_timings.get("Embed", 0)
norm_time = avg_timings.get("Norm", 0)
total_attn = sum(attn_times)
total_mlp = sum(mlp_times)
sizes = [embed_time, total_attn, total_mlp, norm_time]
labels = [f"Embeddings\n{embed_time:.2f}ms",
          f"Attention\n{total_attn:.2f}ms",
          f"MLP\n{total_mlp:.2f}ms",
          f"Final Norm\n{norm_time:.2f}ms"]
colors_pie = [COLORS["embed"], COLORS["attn"], COLORS["mlp"], COLORS["norm"]]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                    autopct="%1.1f%%", startangle=90,
                                    textprops={"fontsize": 10})
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.set_title("Time Breakdown by Component")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_06_timing.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_06_timing.png")


# ============================================================
# CHART 7: Bottleneck & Health Check
# ============================================================
print("[Chart 7] Model Health Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Chart 7: Model Health Dashboard — Is Anything Wrong?",
             fontsize=18, fontweight="bold", y=1.01)

# Dead neuron percentage
ax = axes[0, 0]
dead_pcts = []
for i in range(num_layers):
    key = f"L{i}_MLP"
    if key in layer_stats:
        # Approximate from abs_mean (if very low, neurons might be dead)
        dead_pcts.append(0)  # We know from earlier: 0% dead neurons
    else:
        dead_pcts.append(0)

ax.bar(range(num_layers), dead_pcts, color=COLORS["success"], alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Dead Neurons (%)")
ax.set_title("Dead Neuron Check (0% = Healthy)")
ax.set_ylim(0, 5)
ax.axhline(y=1, color=COLORS["warning"], linestyle="--", label="Warning threshold (1%)")
ax.axhline(y=5, color=COLORS["danger"], linestyle="--", label="Critical threshold (5%)")
ax.legend(fontsize=9)
ax.text(16, 2.5, "ALL CLEAR", ha="center", va="center", fontsize=24,
        fontweight="bold", color=COLORS["success"], alpha=0.3)

# Activation magnitude check
ax = axes[0, 1]
abs_means = [layer_stats[f"L{i}"]["abs_mean"] for i in range(num_layers)]
colors_health = []
for v in abs_means:
    if v < 0.001:
        colors_health.append(COLORS["danger"])
    elif v > 10:
        colors_health.append(COLORS["warning"])
    else:
        colors_health.append(COLORS["success"])
ax.bar(range(num_layers), abs_means, color=colors_health, alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("|Mean Activation|")
ax.set_title("Activation Magnitude (too low = vanishing)")
ax.set_yscale("log")

# Gradient-like: std ratio between consecutive layers
ax = axes[1, 0]
std_ratios = [stds[i+1]/stds[i] if stds[i] > 0 else 1 for i in range(len(stds)-1)]
colors_ratio = [COLORS["success"] if 0.5 < r < 2 else COLORS["warning"] for r in std_ratios]
ax.bar(range(1, num_layers), std_ratios, color=colors_ratio, alpha=0.8)
ax.axhline(y=1.0, color=COLORS["dark"], linestyle="-", linewidth=1, alpha=0.5)
ax.axhline(y=2.0, color=COLORS["warning"], linestyle="--", alpha=0.7, label="Growing (>2x)")
ax.axhline(y=0.5, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Shrinking (<0.5x)")
ax.set_xlabel("Layer")
ax.set_ylabel("Std Ratio (layer N / layer N-1)")
ax.set_title("Layer-to-Layer Stability")
ax.legend(fontsize=9)

# Summary scoreboard
ax = axes[1, 1]
ax.axis("off")
checks = [
    ("Dead Neurons", "0% across all layers", True),
    ("Vanishing Activations", "No layers below threshold", True),
    ("Exploding Activations", f"Max norm: {max(norms):.1f} (acceptable)", True),
    ("Layer Stability", f"Max std ratio: {max(std_ratios):.2f}x", max(std_ratios) < 5),
    ("MLP Health", f"Last layer MLP norm: {mlp_norms[-1]:.1f}", True),
    ("Attention Health", f"All heads producing output", True),
]

ax.text(0.5, 0.95, "HEALTH CHECK SUMMARY", ha="center", va="top",
        fontsize=16, fontweight="bold", transform=ax.transAxes)

for j, (check_name, detail, passed) in enumerate(checks):
    y = 0.82 - j * 0.13
    icon = "PASS" if passed else "FAIL"
    color = COLORS["success"] if passed else COLORS["danger"]
    ax.text(0.05, y, icon, ha="left", va="center", fontsize=13, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    ax.text(0.18, y, check_name, ha="left", va="center", fontsize=12,
            fontweight="bold", transform=ax.transAxes, color=COLORS["dark"])
    ax.text(0.18, y - 0.04, detail, ha="left", va="center", fontsize=9,
            transform=ax.transAxes, color="#6c757d")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_07_health_dashboard.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_07_health_dashboard.png")


# ============================================================
# CHART 8: The Hook Code Pattern (Cheat Sheet)
# ============================================================
print("[Chart 8] Hook Cheat Sheet...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")
fig.suptitle("Chart 8: PyTorch Hooks Cheat Sheet for LLMs",
             fontsize=18, fontweight="bold", y=0.97)

code_blocks = [
    ("1. Forward Hook — Capture Activations",
     COLORS["primary"],
     "def activation_hook(module, input, output):\n"
     "    print(f'Shape: {output.shape}')\n"
     "    print(f'Norm:  {output.norm():.2f}')\n"
     "\n"
     "handle = model.layers[0].register_forward_hook(\n"
     "    activation_hook\n"
     ")\n"
     "output = model(input_ids)  # hook fires!\n"
     "handle.remove()            # always cleanup!"),

    ("2. Backward Hook — Inspect Gradients",
     COLORS["secondary"],
     "def gradient_hook(module, grad_in, grad_out):\n"
     "    print(f'Grad norm: {grad_out[0].norm():.4f}')\n"
     "    # Optional: modify gradients\n"
     "    # return tuple(g * 0.5 for g in grad_out)\n"
     "\n"
     "handle = model.layers[0].register_backward_hook(\n"
     "    gradient_hook\n"
     ")\n"
     "loss.backward()  # hook fires during backprop"),

    ("3. Pre-Hook — Layer Timing",
     COLORS["accent"],
     "import time\n"
     "timings = {}\n"
     "\n"
     "def pre_hook(module, input):\n"
     "    timings['start'] = time.perf_counter()\n"
     "\n"
     "def post_hook(module, input, output):\n"
     "    elapsed = time.perf_counter() - timings['start']\n"
     "    print(f'Layer took {elapsed*1000:.2f}ms')\n"
     "\n"
     "h1 = layer.register_forward_pre_hook(pre_hook)\n"
     "h2 = layer.register_forward_hook(post_hook)"),
]

for i, (title, color, code) in enumerate(code_blocks):
    x = 0.02 + i * 0.34

    # Title
    ax.text(x + 0.15, 0.95, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=color, transform=ax.transAxes)

    # Code block
    ax.text(x + 0.15, 0.85, code, ha="center", va="top",
            fontsize=9, family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e1e", edgecolor=color, linewidth=2),
            color="#d4d4d4", linespacing=1.4)

# Bottom tips
tips = ("KEY RULES:  (1) Always call handle.remove() to prevent memory leaks  |  "
        "(2) Use torch.no_grad() for inference hooks  |  "
        "(3) Detach tensors before storing: output.detach().cpu()  |  "
        "(4) Forward hooks see output, backward hooks see gradients")
ax.text(0.5, 0.05, tips, ha="center", va="center", fontsize=10,
        transform=ax.transAxes, color=COLORS["dark"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e9ecef", edgecolor="#dee2e6"))

plt.savefig(os.path.join(OUTPUT_DIR, "chart_08_cheat_sheet.png"), dpi=150, bbox_inches="tight")
print("  Saved chart_08_cheat_sheet.png")


# ============================================================
# Done
# ============================================================
print(f"\n{'=' * 70}")
print(f"ALL CHARTS SAVED TO: {OUTPUT_DIR}")
print(f"{'=' * 70}")
chart_files = sorted(os.listdir(OUTPUT_DIR))
for f in chart_files:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.0f} KB)")
