# Gemma 4 E4B-it: Complete Architecture Reference

A detailed technical reference for the **Gemma 4 E4B-it** model architecture, extracted directly from the model weights and configuration. All values are verified against the actual HuggingFace checkpoint (`google/gemma-4-E4B-it`).

---

## Overview

| Property | Value |
|----------|-------|
| Model ID | `google/gemma-4-E4B-it` |
| Architecture | `Gemma4ForConditionalGeneration` |
| Total Parameters | **7,996,156,448 (8.00B)** |
| Precision | bfloat16 |
| Context Length | 131,072 tokens |
| Vocabulary Size | 262,144 tokens |
| Modalities | Text, Vision, Audio |

Gemma 4 E4B-it is a **multimodal model** with a text decoder as the backbone, plus a vision encoder and audio encoder for processing images and audio. This document focuses primarily on the **text decoder** since that is what the interpretability notebooks analyze.

---

## High-Level Model Structure

```
Gemma4ForConditionalGeneration
├── model (Gemma4Model)
│   ├── language_model (Gemma4TextModel)          ← Text decoder (main backbone)
│   │   ├── embed_tokens (Gemma4TextScaledWordEmbedding)
│   │   ├── embed_tokens_per_layer (Gemma4TextScaledWordEmbedding)
│   │   ├── per_layer_model_projection (Linear)
│   │   ├── per_layer_projection_norm (Gemma4RMSNorm)
│   │   ├── layers (ModuleList)                   ← 42 decoder layers
│   │   │   ├── Layer 0-4: sliding_attention
│   │   │   ├── Layer 5: full_attention (global)
│   │   │   ├── Layer 6-10: sliding_attention
│   │   │   ├── Layer 11: full_attention (global)
│   │   │   ├── ... (repeating 5:1 pattern)
│   │   │   └── Layer 41: full_attention (global)
│   │   ├── norm (Gemma4RMSNorm)                  ← Final RMSNorm
│   │   └── rotary_emb (Gemma4TextRotaryEmbedding)
│   ├── vision_tower (Gemma4VisionModel)          ← Vision encoder
│   │   ├── patch_embedder
│   │   ├── encoder (16 layers)
│   │   └── pooler
│   ├── embed_vision (Gemma4MultimodalEmbedder)   ← Vision → text projection
│   ├── audio_tower (Gemma4AudioModel)            ← Audio encoder
│   │   ├── subsample_conv_projection
│   │   ├── rel_pos_enc
│   │   ├── layers (12 layers)
│   │   └── output_proj
│   └── embed_audio (Gemma4MultimodalEmbedder)    ← Audio → text projection
└── lm_head (Linear)                              ← Output projection (tied weights)
```

---

## Text Decoder Configuration

### Core Dimensions

| Parameter | Value |
|-----------|-------|
| Hidden size | 2,560 |
| Number of layers | 42 |
| Intermediate size (MLP) | 10,240 (4x hidden) |
| Head dimension (sliding) | 256 |
| Head dimension (global) | 512 |
| Vocabulary size | 262,144 |
| Max position embeddings | 131,072 |
| RMS norm epsilon | 1e-6 |
| Hidden activation | `gelu_pytorch_tanh` |
| Attention bias | False |
| Tied word embeddings | True |

### Per-Layer Input Processing

Gemma 4 introduces a unique **per-layer input gate** mechanism — each decoder layer has an auxiliary pathway that re-embeds the input tokens into a smaller (256-dim) space and mixes it with the residual stream:

| Component | Shape | Parameters |
|-----------|-------|------------|
| `embed_tokens_per_layer` | (262144, 256) | Shared across layers |
| `per_layer_model_projection` | (256, 2560) → (2560, 256) | Shared projection |
| `per_layer_input_gate` | (256, 2560) | Per-layer, 655,360 params |
| `per_layer_projection` | (2560, 256) | Per-layer, 655,360 params |
| `post_per_layer_input_norm` | (2560,) | Per-layer RMSNorm |

This means each layer gets a "fresh look" at the raw token embeddings, gated and projected, added into the residual stream. This is a distinctive Gemma 4 innovation not found in Llama, Mistral, or earlier Gemma models.

---

## Hybrid Attention: Sliding Window + Global

Gemma 4's most distinctive architectural feature is its **hybrid attention pattern**. Instead of uniform attention across all layers, it alternates between two types:

### Attention Pattern (5:1 Ratio)

```
Layer  0: sliding_attention    ─┐
Layer  1: sliding_attention     │
Layer  2: sliding_attention     ├── Block 1 (5 sliding + 1 global)
Layer  3: sliding_attention     │
Layer  4: sliding_attention     │
Layer  5: full_attention       ─┘
Layer  6: sliding_attention    ─┐
Layer  7: sliding_attention     │
Layer  8: sliding_attention     ├── Block 2
Layer  9: sliding_attention     │
Layer 10: sliding_attention     │
Layer 11: full_attention       ─┘
Layer 12-17: Block 3 ...
Layer 18-23: Block 4 ...
Layer 24-29: Block 5 ...
Layer 30-35: Block 6 ...
Layer 36-41: Block 7 (final block, layer 41 is global)
```

**7 global attention layers** at positions: 5, 11, 17, 23, 29, 35, 41
**35 sliding window layers** at all other positions

### Sliding Window Attention

| Property | Value |
|----------|-------|
| Window size | **512 tokens** |
| Query heads | 8 |
| KV heads | 2 |
| Head dimension | 256 |
| GQA group size | 4 (each KV head serves 4 query heads) |
| RoPE | Default, theta=10,000 |

Each token can only attend to the **512 preceding tokens** (and itself). This makes sliding window layers O(n) in memory for long sequences.

### Global (Full) Attention

| Property | Value |
|----------|-------|
| Window size | Full context (131,072) |
| Query heads | **16** |
| KV heads | **4** |
| Head dimension | **512** |
| GQA group size | 4 (each KV head serves 4 query heads) |
| RoPE | Proportional, theta=1,000,000, partial_rotary_factor=0.25 |

Global layers have **doubled** head dimensions (512 vs 256) and **doubled** head counts (16q/4kv vs 8q/2kv), resulting in **4x the attention parameters** compared to sliding layers. Each token can attend to **all** preceding tokens.

### Attention Parameter Comparison

| Component | Sliding Layer | Global Layer |
|-----------|--------------|--------------|
| q_proj | (2048, 2560) = 5.2M | (4096, 2560) = 10.5M |
| k_proj | (512, 2560) = 1.3M | (1024, 2560) = 2.6M |
| v_proj | (512, 2560) = 1.3M | (1024, 2560) = 2.6M |
| o_proj | (2560, 2048) = 5.2M | (2560, 4096) = 10.5M |
| q_norm | (256,) | (512,) |
| k_norm | (256,) | (512,) |
| **Total attention** | **13.1M** | **26.2M** |

### RoPE (Rotary Position Embeddings)

Gemma 4 uses **different RoPE configurations** for each attention type:

- **Sliding attention**: Standard RoPE with `theta=10,000`
- **Global attention**: Proportional RoPE with `theta=1,000,000` and `partial_rotary_factor=0.25` (only 25% of dimensions get rotary encoding)

The high theta value and partial rotation in global layers enables better long-range position discrimination over the full 131K context.

---

## Decoder Layer Anatomy

Each of the 42 decoder layers has the same structural template, but with different parameter sizes depending on whether it's a sliding or global attention layer.

### Sliding Window Layer (35 layers)

```
Input (residual stream, 2560-dim)
  │
  ├─── per_layer_input_gate(embed_tokens_per_layer(tokens))  ← Fresh token signal
  │
  ├── input_layernorm (RMSNorm)
  │     └── Self-Attention (8q/2kv, head_dim=256, window=512)
  │           ├── q_proj: (2560) → (2048)     [8 heads x 256]
  │           ├── k_proj: (2560) → (512)      [2 heads x 256]
  │           ├── v_proj: (2560) → (512)      [2 heads x 256]
  │           ├── q_norm, k_norm (RMSNorm, 256-dim)
  │           ├── RoPE (theta=10000)
  │           ├── GQA attention (each KV head → 4 Q heads)
  │           └── o_proj: (2048) → (2560)
  ├── post_attention_layernorm (RMSNorm)
  │     └── + residual
  │
  ├── pre_feedforward_layernorm (RMSNorm)
  │     └── MLP (Gated)
  │           ├── gate_proj: (2560) → (10240)
  │           ├── up_proj:   (2560) → (10240)
  │           ├── GELU activation on gate
  │           ├── element-wise multiply: gate * up
  │           └── down_proj: (10240) → (2560)
  ├── post_feedforward_layernorm (RMSNorm)
  │     └── + residual
  │
  └── post_per_layer_input_norm (RMSNorm)
        └── Output (residual stream, 2560-dim)
```

**Parameters per sliding layer: ~92.5M**

### Global Attention Layer (7 layers)

Same structure, but attention sub-layer has doubled dimensions:

```
  Self-Attention (16q/4kv, head_dim=512, full context)
    ├── q_proj: (2560) → (4096)     [16 heads x 512 — note: not 256]
    ├── k_proj: (2560) → (1024)     [4 heads x 512]
    ├── v_proj: (2560) → (1024)     [4 heads x 512]
    ├── q_norm, k_norm (RMSNorm, 512-dim)
    ├── RoPE (theta=1M, partial_rotary_factor=0.25)
    └── o_proj: (4096) → (2560)
```

**Parameters per global layer: ~105.6M**

### Four Normalization Layers Per Block

Unlike Llama (which uses 2 norms per layer: pre-attention and pre-MLP), Gemma 4 uses **4 RMSNorm layers** per decoder block:

1. `input_layernorm` — before attention
2. `post_attention_layernorm` — after attention, before residual add
3. `pre_feedforward_layernorm` — before MLP
4. `post_feedforward_layernorm` — after MLP, before residual add

This "sandwich" normalization (pre + post for both sub-layers) provides more stable training dynamics.

---

## Layer-by-Layer Projection Shapes

### Complete Layer Map

| Layer | Type | Q proj | K proj | V proj | O proj | Q heads | KV heads | Head dim |
|-------|------|--------|--------|--------|--------|---------|----------|----------|
| 0 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 1 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 2 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 3 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 4 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **5** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 6 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 7 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 8 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 9 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| 10 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **11** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 12-16 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **17** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 18-22 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **23** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 24-28 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **29** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 30-34 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **35** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |
| 36-40 | sliding | (2048, 2560) | (512, 2560) | (512, 2560) | (2560, 2048) | 8 | 2 | 256 |
| **41** | **global** | **(4096, 2560)** | **(1024, 2560)** | **(1024, 2560)** | **(2560, 4096)** | **16** | **4** | **512** |

---

## MLP (Feed-Forward Network)

All 42 layers share the same MLP structure — a **gated MLP** with GELU activation:

```
x → gate_proj(x) → GELU → element_multiply ← up_proj(x)
                                    │
                              down_proj → output
```

| Component | Shape | Parameters |
|-----------|-------|------------|
| gate_proj | (10240, 2560) | 26,214,400 |
| up_proj | (10240, 2560) | 26,214,400 |
| down_proj | (2560, 10240) | 26,214,400 |
| **Total per layer** | | **78,643,200** |

The intermediate size is **4x the hidden size** (10240 = 2560 x 4). The gated architecture uses two parallel projections (gate + up) multiplied together before the down projection, following the SwiGLU/GeGLU pattern.

---

## Embedding & Output Layers

### Token Embedding

| Component | Shape | Parameters |
|-----------|-------|------------|
| `embed_tokens` | (262144, 2560) | 671,088,640 |
| Type | `Gemma4TextScaledWordEmbedding` | Scaled by sqrt(hidden_size) |

The embedding output is **scaled by sqrt(2560) = 50.6** before being fed to the first decoder layer. This is a Gemma-specific design choice that differs from most other LLMs.

### Output Head

| Component | Shape | Parameters |
|-----------|-------|------------|
| `final_norm` | (2560,) | 2,560 (RMSNorm) |
| `lm_head` | (262144, 2560) | **Tied to embed_tokens** |
| `final_logit_softcapping` | 30.0 | Caps logits via tanh |

**Weight tying**: The `lm_head` shares weights with `embed_tokens`, so the output projection does not add extra parameters.

**Logit softcapping**: Final logits are passed through `30.0 * tanh(logits / 30.0)`, which smoothly caps extreme values at +/-30. This is a Gemma innovation for training stability.

---

## Grouped Query Attention (GQA)

Gemma 4 uses GQA to reduce KV cache memory while maintaining model quality.

### Sliding Window Layers
```
Query heads:  [Q0] [Q1] [Q2] [Q3]  |  [Q4] [Q5] [Q6] [Q7]
                  ↓                        ↓
KV heads:        [KV0]                    [KV1]
Group size:    4 query heads per KV head
```

### Global Attention Layers
```
Query heads:  [Q0][Q1][Q2][Q3] | [Q4][Q5][Q6][Q7] | [Q8][Q9][Q10][Q11] | [Q12][Q13][Q14][Q15]
                    ↓                   ↓                    ↓                      ↓
KV heads:         [KV0]               [KV1]                [KV2]                  [KV3]
Group size:    4 query heads per KV head
```

### KV Cache Savings

| Attention Type | KV heads | Head dim | KV cache per token | vs Full attention |
|---------------|----------|----------|-------------------|-------------------|
| Sliding (8q/2kv) | 2 | 256 | 1,024 bytes (bf16) | 4x reduction |
| Global (16q/4kv) | 4 | 512 | 4,096 bytes (bf16) | 4x reduction |

---

## Query and Key Normalization

Gemma 4 applies **RMSNorm to Q and K vectors** after projection but before RoPE:

```
Q = q_norm(q_proj(x))    # RMSNorm on (head_dim,) per head
K = k_norm(k_proj(x))    # RMSNorm on (head_dim,) per head
Q = apply_rope(Q)
K = apply_rope(K)
```

| Layer type | q_norm dim | k_norm dim |
|-----------|-----------|-----------|
| Sliding | (256,) | (256,) |
| Global | (512,) | (512,) |

This QK normalization prevents attention logit explosion during training and is shared across all heads within a layer.

---

## Parameter Budget Breakdown

### By Component Type

| Component | Count | Params per unit | Total | % of model |
|-----------|-------|-----------------|-------|------------|
| Embedding (shared with lm_head) | 1 | 671M | 671M | 8.4% |
| Sliding attention layers | 35 | 13.1M | 459M | 5.7% |
| Global attention layers | 7 | 26.2M | 183M | 2.3% |
| MLP layers | 42 | 78.6M | 3,303M | 41.3% |
| Layer norms (4 per layer + final) | 169 | 2,560 | 0.4M | <0.1% |
| Per-layer input gates | 42 | 1.3M | 55M | 0.7% |
| Per-layer embeddings & projection | 1 | ~672M | 672M | 8.4% |
| Vision tower | 1 | ~151M | 151M | 1.9% |
| Audio tower | 1 | ~42M | 42M | 0.5% |
| Multimodal embedders | 2 | ~4M | 8M | 0.1% |
| **Total** | | | **~8.00B** | **100%** |

### Key Insight

The **MLP layers dominate** the parameter budget at 41.3%, while attention accounts for only 8.0% total. This is because the intermediate size (10240) is 4x the hidden size, and there are 42 layers. The vision and audio components together are only ~2.4% of the model.

---

## Vision Tower

The vision encoder processes images into embeddings that are projected into the text decoder's hidden space.

| Property | Value |
|----------|-------|
| Architecture | `Gemma4VisionModel` |
| Hidden size | 768 |
| Layers | 16 |
| Attention heads | 12 |
| KV heads | 12 (MHA, not GQA) |
| Head dimension | 64 |
| Intermediate size | 3,072 |
| Patch size | 16x16 pixels |
| Pooling kernel | 3 |
| Soft tokens per image | 280 |
| Position embedding size | 10,240 |

Each image is converted to 280 soft tokens that are projected to 2560-dim and inserted into the text sequence.

---

## Audio Tower

The audio encoder processes audio waveforms into embeddings.

| Property | Value |
|----------|-------|
| Architecture | `Gemma4AudioModel` |
| Hidden size | 1,024 |
| Layers | 12 |
| Attention heads | 8 |
| Attention chunk size | 12 |
| Context (left/right) | 13 / 0 (causal) |
| Conv kernel size | 5 |
| Output projection | 1,536-dim |
| Activation | SiLU |

---

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<pad>` | 0 | Padding |
| `<eos>` | 1 | End of sequence |
| `<bos>` | 2 | Beginning of sequence |
| Turn marker | 106 | Also treated as EOS |
| `<image>` | 258880 | Image placeholder |
| `<boi>` | 255999 | Begin of image |
| `<eoi>` | 258882 | End of image |
| `<audio>` | 258881 | Audio placeholder |
| `<boa>` | 256000 | Begin of audio |
| `<eoa>` | 258883 | End of audio |
| `<video>` | 258884 | Video placeholder |

---

## Hook Access Paths (PyTorch)

For mechanistic interpretability work, here are the key module paths:

```python
# Loading
from transformers import AutoProcessor, AutoModelForCausalLM
processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it",
    dtype="auto", device_map="auto")

# Embeddings
model.model.language_model.embed_tokens          # Token embedding
model.model.language_model.embed_tokens_per_layer # Per-layer embedding

# Decoder layers (i = 0..41)
model.model.language_model.layers[i]                    # Full layer
model.model.language_model.layers[i].self_attn           # Attention module
model.model.language_model.layers[i].self_attn.q_proj    # Query projection
model.model.language_model.layers[i].self_attn.k_proj    # Key projection
model.model.language_model.layers[i].self_attn.v_proj    # Value projection
model.model.language_model.layers[i].self_attn.o_proj    # Output projection
model.model.language_model.layers[i].self_attn.q_norm    # Query RMSNorm
model.model.language_model.layers[i].self_attn.k_norm    # Key RMSNorm
model.model.language_model.layers[i].mlp                 # MLP module
model.model.language_model.layers[i].mlp.gate_proj       # Gate projection
model.model.language_model.layers[i].mlp.up_proj         # Up projection
model.model.language_model.layers[i].mlp.down_proj       # Down projection
model.model.language_model.layers[i].input_layernorm     # Pre-attention norm
model.model.language_model.layers[i].post_attention_layernorm
model.model.language_model.layers[i].pre_feedforward_layernorm
model.model.language_model.layers[i].post_feedforward_layernorm
model.model.language_model.layers[i].per_layer_input_gate  # Per-layer gate

# Final layers
model.model.language_model.norm     # Final RMSNorm
model.lm_head                       # Output projection (tied to embed_tokens)

# Vision & Audio
model.model.vision_tower            # Vision encoder
model.model.audio_tower             # Audio encoder
```

---

## Comparison with Other Models

| Feature | Gemma 4 E4B | Llama 3.1 8B | Mistral 7B | Gemma 2 9B |
|---------|-------------|-------------|------------|------------|
| Parameters | 8.0B | 8.0B | 7.2B | 9.2B |
| Layers | 42 | 32 | 32 | 42 |
| Hidden size | 2,560 | 4,096 | 4,096 | 3,584 |
| MLP intermediate | 10,240 | 14,336 | 14,336 | 14,336 |
| Attention | Hybrid (sliding+global) | Full global | Sliding window | Hybrid |
| Sliding window | 512 tokens | N/A | 4,096 tokens | 4,096 tokens |
| Q heads (sliding) | 8 | 32 | 32 | 16 |
| KV heads (sliding) | 2 | 8 | 8 | 8 |
| Q heads (global) | 16 | N/A | N/A | 16 |
| KV heads (global) | 4 | N/A | N/A | 8 |
| Head dim | 256/512 | 128 | 128 | 256 |
| Context length | 131K | 128K | 32K | 8K |
| Vocab size | 262,144 | 128,256 | 32,000 | 256,000 |
| Norms per layer | 4 (sandwich) | 2 (pre-norm) | 2 (pre-norm) | 4 (sandwich) |
| QK norm | Yes | No | No | Yes |
| Logit softcapping | Yes (30.0) | No | No | Yes (30.0) |
| Embedding scaling | Yes (sqrt) | No | No | Yes (sqrt) |
| Per-layer input gate | Yes | No | No | No |
| Tied embeddings | Yes | No | No | Yes |
| Modalities | Text+Vision+Audio | Text | Text | Text |

---

## Key Architectural Innovations

### 1. Per-Layer Input Gates
Each decoder layer gets a "shortcut" connection from the raw token embeddings through a gated projection. This allows deeper layers to maintain access to token-level information without relying solely on the residual stream — potentially reducing the "forgetting" problem in deep networks.

### 2. Asymmetric Hybrid Attention
The 5:1 sliding-to-global ratio with **different head dimensions** per type is unique. Global layers don't just have wider windows — they have fundamentally larger attention capacity (512-dim heads vs 256-dim), allowing them to capture more complex long-range dependencies.

### 3. Sandwich Normalization
Four norms per layer (pre+post for both attention and MLP) provides more gradient control points. This, combined with QK normalization, enables stable training at scale.

### 4. Massive Vocabulary
At 262,144 tokens, Gemma 4's vocabulary is 2x Llama's and 8x Mistral's. Combined with the per-layer embedding pathway, this suggests the model encodes more semantic information directly in the tokenization.

---

## Configuration Values Reference

The complete `text_config` dump for programmatic access:

```python
{
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2560,
    "hidden_size_per_layer_input": 256,
    "intermediate_size": 10240,
    "max_position_embeddings": 131072,
    "num_attention_heads": 8,
    "num_hidden_layers": 42,
    "num_key_value_heads": 2,
    "num_kv_shared_layers": 18,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default"
        }
    },
    "sliding_window": 512,
    "tie_word_embeddings": True,
    "vocab_size": 262144,
}
```

---

## References

- Google DeepMind (2025). *Gemma 4 Technical Report*. [HuggingFace Model Card](https://huggingface.co/google/gemma-4-E4B-it)
- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Shazeer, N. (2019). [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150). (Multi-Query / Grouped-Query Attention)
- Ainslie, J. et al. (2023). [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). EMNLP 2023.
- Su, J. et al. (2021). [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864). (RoPE)
- Zhang, B. & Sennrich, R. (2019). [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467). NeurIPS 2019. (RMSNorm)
- Shazeer, N. (2020). [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202). (Gated MLP / SwiGLU)
- Child, R., Gray, S., Radford, A., Sutskever, I. (2019). [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509). (Sliding window attention)
- Beltagy, I., Peters, M.E., Cohan, A. (2020). [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150). (Hybrid local+global attention)
