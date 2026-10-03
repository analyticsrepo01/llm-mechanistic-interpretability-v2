# Advanced LLM Activation Analysis Techniques
## Research Reference for PyTorch Hook-Based Implementation on Llama 3.1 8B

---

## Techniques Overview

| # | Technique | Reveals | Difficulty | Notebook |
|---|-----------|---------|------------|----------|
| 1 | Logit Lens / Tuned Lens | When predictions form layer by layer | Easy | 03_logit_lens.ipynb |
| 2 | Outlier Feature Detection | Dominant hidden dimensions | Easy | 04_outlier_features.ipynb |
| 3 | CKA Layer Similarity | Layer phase structure, redundancy | Easy-Medium | 05_layer_redundancy_cka.ipynb |
| 4 | Token Trajectory | Per-token representation evolution | Easy-Medium | 06_token_trajectory.ipynb |
| 5 | Direct Logit Attribution | Per-component prediction contribution | Medium | 07_direct_logit_attribution.ipynb |
| 6 | Representation Geometry (PCA/UMAP) | How model organizes concepts | Easy | 08_representation_geometry.ipynb |
| 7 | Activation Patching | Causal importance of locations | Medium-Hard | Future |
| 8 | Activation Steering | Concept directions, behavioral control | Medium | Future |
| 9 | Linear Probes | What info each layer encodes | Medium | Future |
| 10 | Sparse Autoencoders | Monosemantic features | Hard | Future |
| 11 | Attention Head Classification | Head function types (induction, etc.) | Medium | Future |
| 12 | Knowledge Neurons | Factual storage locations | Medium-Hard | Future |
| 13 | Function Vectors | How in-context learning works | Medium | Future |
| 14 | Information Flow / Knockout | Circuit discovery | Hard | Future |
| 15 | Layer Redundancy (Skip-layer) | Prunable layers | Easy | 05_layer_redundancy_cka.ipynb |

---

## 1. Logit Lens / Tuned Lens

**Paper:** nostalgebraist, "interpreting GPT: the logit lens" (2020); Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023)

**What it reveals:** Shows what the model would predict if decoding happened at each intermediate layer. Reveals how token predictions form and refine layer by layer -- watching a vague semantic representation sharpen into a confident prediction.

**Implementation:** Apply the model's final LayerNorm + LM head to each layer's hidden state output. The "Tuned Lens" variant trains a small affine transform per layer for cleaner results.

**Visualization:** Heatmap where x-axis = token position, y-axis = layer, color/text = top predicted token.

---

## 2. Outlier Feature Detection

**Paper:** Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (NeurIPS 2022)

**What it reveals:** Certain hidden dimensions develop extremely large activation magnitudes (100-1000x larger than typical). These dominate computation and are critical for model performance -- quantizing them destroys quality.

**Implementation:** Per-dimension magnitude statistics across batch and sequence. Flag dimensions > 6 std above mean.

**Key finding for Llama:** Outlier features emerge in layers 6+ and concentrate in specific, consistent hidden dimensions (typically 6-10 extreme outliers).

**Visualization:** Heatmap (hidden dim x layer) showing magnitude, with bright vertical stripes at outlier dims.

---

## 3. CKA Layer Similarity

**Paper:** Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019); Men et al., "ShortGPT: Layers in LLMs are More Redundant Than You Expect" (2024)

**What it reveals:** Quantifies representation similarity between layers. Reveals "phase structure" -- groups of layers doing similar work vs. dramatic transitions. Identifies redundant layers for pruning.

**Implementation:** Linear CKA using HSIC (Hilbert-Schmidt Independence Criterion). Compute pairwise for all 32x32 layer pairs.

**Visualization:** 32x32 heatmap showing block-diagonal structure. Also: importance bar chart per layer.

---

## 4. Token Trajectory (Residual Stream Analysis)

**Paper:** Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021)

**What it reveals:** Tracks how individual token representations change through the residual stream. Shows when the model "decides" what a token means in context. Decomposes contributions from attention vs. MLP at each layer.

**Implementation:** Hook each layer's output and track a specific token position. Compute cosine similarity between consecutive layers.

**Visualization:** Cosine similarity line plot (drops = high-change layers). Stacked bar chart of attention vs. MLP contribution.

---

## 5. Direct Logit Attribution (DLA)

**Paper:** Nanda, "A Comprehensive Mechanistic Interpretability Explainer" (2022); derived from residual stream framework

**What it reveals:** Decomposes the final logit into contributions from each component (each layer's attention + MLP). Answers: "which layer was most responsible for predicting this token?"

**Implementation:** Due to the residual stream, final_logit = sum of all component contributions projected onto the unembedding vector.

**Visualization:** Waterfall/stacked bar chart showing positive (promoting) and negative (suppressing) contributions per component.

---

## 6. Representation Geometry (PCA/UMAP)

**Paper:** General technique. Applied to LLMs by various works including Reif et al., "Visualizing and Measuring the Geometry of BERT" (NeurIPS 2019)

**What it reveals:** How the model organizes concepts in representation space. Semantic categories form clusters. Shows how representations change across layers (early: mixed; late: separated).

**Implementation:** Collect hidden states from multiple prompts, apply PCA or UMAP for dimensionality reduction.

**Visualization:** 2D scatter plots colored by category, one per layer. Animated progression is compelling.

---

## 7. Activation Patching (Causal Tracing)

**Paper:** Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, NeurIPS 2022)

**What it reveals:** Identifies which layers and token positions are causally responsible for a behavior. Run clean prompt, corrupt prompt, then patch clean activations back one location at a time.

**Visualization:** 2D heatmap (token pos x layer) of causal effect strength.

---

## 8. Activation Steering / Representation Engineering

**Paper:** Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023); Turner et al., "Activation Addition" (2023)

**What it reveals:** Specific directions in activation space correspond to high-level concepts (truthfulness, sentiment, refusal). Adding/subtracting steering vectors controls behavior.

**Implementation:** Compute contrastive mean difference between positive/negative example sets. Apply via hook at mid-to-late layers.

---

## 9. Linear Probes

**Paper:** Alain & Bengio, "Understanding Intermediate Layers using Linear Classifier Probes" (2016); Belinkov, "Probing Classifiers" (2022)

**What it reveals:** What information each layer encodes (syntax, semantics, facts). Train simple linear classifier on hidden states to predict properties.

---

## 10. Sparse Autoencoders (SAEs)

**Paper:** Bricken et al., "Towards Monosemanticity" (Anthropic, 2023); Templeton et al., "Scaling Monosemanticity" (Anthropic, 2024)

**What it reveals:** Decomposes polysemantic neurons into interpretable monosemantic features.

---

## 11. Attention Head Classification

**Paper:** Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022)

**What it reveals:** Different heads serve different roles: induction heads, previous-token heads, name movers, etc.

---

## 12. Knowledge Neurons

**Paper:** Dai et al., "Knowledge Neurons in Pretrained Transformers" (ACL 2022)

**What it reveals:** Specific MLP neurons store factual knowledge. Suppressing them removes the fact.

---

## 13. Function Vectors

**Paper:** Todd et al., "Function Vectors in Large Language Models" (ICLR 2024)

**What it reveals:** ICL creates a "function vector" in the residual stream encoding the task mapping.

---

## 14. Information Flow / Attention Knockout

**Paper:** Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" (2023)

**What it reveals:** Traces information flow paths through the model to discover circuits.

---

## 15. Layer Redundancy (ShortGPT)

**Paper:** Men et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" (2024)

**What it reveals:** Many LLM layers are surprisingly redundant. Uses Block Importance (BI) score via cosine similarity between layer input and output.

---

# 2025-2026 Cutting-Edge Research

| # | Technique | Paper/Source | Date | Notebook |
|---|-----------|-------------|------|----------|
| 16 | Persona Vectors | Anthropic Research | Aug 2025 | 09_persona_vectors.ipynb |
| 17 | Activation Steering with Emotion Concepts | Anthropic Research | Apr 2026 | 10_emotion_steering.ipynb |
| 18 | KV Cache SAEs | arXiv | Dec 2025 | Future |
| 19 | SALVE (SAE + Model Editing) | ICLR 2026 | Dec 2025 | Future |
| 20 | ACC++ (Prompt-Specific Circuits) | arXiv | Feb 2026 | Future |
| 21 | Introspection / Concept Injection | Anthropic Research | Oct 2025 | Future |

---

## 16. Persona Vectors (Activation Steering)

**Paper:** Anthropic, "Persona Vectors: Personality in the Weights" (Aug 2025)

**What it reveals:** Personality traits (helpfulness, honesty, formality, etc.) are encoded as linear directions in activation space. By computing the mean activation difference between opposing behavioral prompts, you can extract a "persona vector" that causally controls behavior when injected during inference.

**Key insight:** Tested directly on **Llama-3.1-8B-Instruct** (our exact model!). Persona vectors are extractable from mid-to-late layers (layers 12-24 most effective). Steering strength is tunable — multiply the vector by a scalar to control intensity.

**Implementation:**
1. Create contrastive prompt pairs (e.g., "You are extremely helpful" vs "You are unhelpful and dismissive")
2. Run both through model, capture hidden states at each layer using forward hooks
3. Compute mean difference vector: `persona_vector = mean(helpful_activations) - mean(unhelpful_activations)`
4. At inference, add `alpha * persona_vector` to the residual stream at target layers
5. Measure behavioral change via generation quality metrics

**Visualization:** Cosine similarity heatmap of persona vectors across layers, PCA of steered vs unsteered activations, generation comparison panels.

---

## 17. Emotion Concepts (Causal Activation Steering)

**Paper:** Anthropic, "Emotion Concepts in Language Models" (Apr 2026)

**What it reveals:** 171 emotion vectors extracted from model activations using contrastive pairs. Emotions are encoded as linear directions. Key finding: **desperation vectors causally increase unethical behavior** — demonstrating that internal "emotional" representations have behavioral consequences.

**Key insight:** Builds on persona vectors but focuses on emotional states rather than personality traits. Emotion vectors cluster into interpretable families (positive/negative, high/low arousal). The causal relationship between emotional activation and behavioral outcomes is a breakthrough in AI safety.

**Implementation:**
1. Define emotion pairs (calm/anxious, honest/deceptive, confident/desperate)
2. Generate contrastive prompts that elicit each emotion
3. Extract mean difference vectors from target layers
4. Steer the model by injecting emotion vectors and measure behavioral changes
5. Test with ethical dilemma prompts to verify causal influence

**Visualization:** Emotion vector cluster map (PCA/UMAP), steering strength curves, behavioral outcome matrices.

---

## 18. KV Cache SAEs (Semantic Atoms)

**Paper:** "Decomposing the KV Cache with Sparse Autoencoders" (Dec 2025)

**What it reveals:** Top-K sparse autoencoders can decompose the KV cache into interpretable "semantic atoms." Key discovery: **keys are sparse routers** (selecting which information to attend to) while **values are dense content payloads** (carrying the actual information). This key-value asymmetry was previously unknown.

**Implementation:** Train Top-K SAEs on captured key and value cache tensors separately. Analyze sparsity patterns and feature interpretability of learned dictionary elements.

---

## 19. SALVE (Unified SAE Framework)

**Paper:** "SALVE: Discover, Validate, Control with SAEs" (Dec 2025, ICLR 2026)

**What it reveals:** A unified three-step framework: (1) Discover interpretable features via SAEs, (2) Validate them with causal interventions, (3) Control model behavior by manipulating features. Bridges the gap between feature discovery and practical model editing.

---

## 20. ACC++ (Prompt-Specific Circuits)

**Paper:** "ACC++: Automated Circuit Discovery" (Feb 2026)

**What it reveals:** Circuits in LLMs are **prompt-specific**, not task-specific. Different prompts for the same task activate different circuits. Prompts cluster into "prompt families" that share circuit structure. No SAEs or activation patching required — uses gradient-based attribution.

---

## 21. Introspection / Concept Injection

**Paper:** Anthropic, "Can Models Report on Their Own Internal States?" (Oct 2025)

**What it reveals:** Tests whether LLMs can accurately report on their internal representations. Uses concept injection (adding known vectors to activations) to test detection rate. Result: ~20% detection rate on Claude Opus 4.1, suggesting limited but non-trivial introspective capability.
