# Solar-Open-100B vs GLM-4.5-Air: Weight Provenance Analysis

A reproduction workspace for investigating claims about whether Upstage's [Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) was derived from Zhipu's [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air).

## Background

**2026-01-01** — [Sionic AI published an analysis](https://github.com/sionic-ai/solar-vs-glm) claiming Solar-Open-100B was derived from GLM-4.5-Air, not trained from scratch. Key evidence:
- LayerNorm/RMSNorm cosine similarity of ~0.99 between same-layer parameters across models
- Within-model (different layers) baseline showed only ~0.38 similarity
- Claimed this "182 sigma" deviation proves weight initialization from GLM

**2026-01-01** (6 hours later) — [Hyunwoong Ko published a rebuttal](https://github.com/hyunwoongko/solar-vs-glm-vs-phi) arguing the methodology was flawed:
- Within-model similarity for layers 10/20/30 is actually ~0.99 (Sionic only tested layer 0 vs others)
- Layer 0's `input_layernorm` is special (receives raw embeddings) and behaves differently
- Phi-3.5-MoE shows similarly high cross-model cosine with GLM despite being unrelated
- Centered cosine similarity (mean-subtracted) drops cross-model similarity to ~0
- High raw cosine reflects RMSNorm initialization bias (ones-init), not derivation

## The Methodological Dispute

| Claim | Sionic (accusation) | Hyunwoong Ko (rebuttal) |
|-------|---------------------|-------------------------|
| Within-model LayerNorm similarity | ~0.38 (layer 0 vs 10/20/30/40) | ~0.99 (layer 10 vs 20 vs 30) |
| Cross-model similarity | ~0.99 (proves derivation) | ~0.99 (proves nothing—Phi also high) |
| Key metric | Raw cosine similarity | Centered cosine / Pearson (drops to ~0) |

**Core issue:** RMSNorm weights are initialized to 1.0 and maintain low variance during training. Raw cosine similarity on such vectors is dominated by this shared initialization bias, not meaningful structural similarity.

## Architecture Differences

Despite the LayerNorm similarity, the models differ substantially:

| Parameter | GLM-4.5-Air | Solar-Open-100B |
|-----------|-------------|-----------------|
| num_hidden_layers | 46 | 48 |
| num_attention_heads | 96 | 64 |
| vocab_size | 151,552 | 196,608 |

Attention projections (`q_proj`, `o_proj`) have different shapes due to head count differences. Only `k_proj`, `v_proj`, and MoE routers are shape-compatible.

## This Repository

We created a comprehensive plan to independently verify the claims using multiple fingerprinting techniques:

- **[PLAN.md](PLAN.md)** — Full experimental protocol with 10+ experiments
- **[reference/](reference/)** — Archived source repos (`.tar.zst`) and relevant papers (`.pdf`)
- **[ANALYSIS-*.md](.)** — LLM-generated summaries of the dispute (for context, not ground truth)

### Key Experiments in the Plan

1. Centered cosine / Pearson correlation (removes mean bias)
2. Byte-identity tests (strongest evidence if positive)
3. Layer alignment matrices with margins
4. Higher-dimensional tensor comparison (`k_proj`, `v_proj`, routers)
5. CKA fingerprinting (AWM-style)
6. Intrinsic std-curve fingerprints (LLM-Fingerprint style)

## Discussion

- [r/LocalLLaMA: Upstage Solar-Open-100B public validation](https://www.reddit.com/r/LocalLLaMA/comments/1q0zst6/upstage_solaropen100b_public_validation/) — community discussion with methodological feedback

## References

See [PLAN.md § Reference Repos](PLAN.md#9-reference-repos-local-copies) for full list of archived sources and papers.

## Status

This is a fingerprinting exercise to evaluate the methodology, not a definitive verdict. The rebuttal raises valid concerns about raw cosine similarity on LayerNorm vectors. Stronger evidence would require:
- Byte-identical tensor matches, or
- Diagonal-dominant alignment on high-dimensional tensors with significant margins over controls
