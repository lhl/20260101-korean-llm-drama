# Repro Plan: Solar‑Open‑100B vs GLM‑4.5‑Air (weight provenance)

This workspace contains:
- `sionic-ai--solar-vs-glm/` (Range‑request based tensor probing; no full shard downloads needed)
- `hyunwoongko--solar-vs-glm-vs-phi/` (rebuttal; loads full models via `transformers`, which is usually impractical for 100B‑class weights)
- `ANALYSIS-*.md` (LLM summaries of the dispute; useful for context, not ground truth)

The goal of this plan is to produce **reproducible, CPU‑friendly** experiments that let us verify (or falsify) the core claims ourselves.

---

## 0) What we’re trying to answer

We want to test a spectrum of hypotheses:

1. **Direct weight reuse**: some tensors are *identical* between `upstage/Solar-Open-100B` and `zai-org/GLM-4.5-Air`.
2. **Initialized-from**: Solar started from GLM weights then continued training / modified architecture.
3. **Independent training**: similarities are due to convergent architecture choices + norm parameter behavior (cosine traps), not shared initialization.

**Important:** LayerNorm/RMSNorm vectors are low‑information and often biased positive; raw cosine similarity can be misleading. Prefer stronger tests (byte identity, permutation/alignment, higher‑dimensional tensors, controls, and “centered” metrics).

---

## 1) Models to use

Primary models:
- `upstage/Solar-Open-100B` (bf16, 48 layers, hidden 4096, MoE 128 experts)
- `zai-org/GLM-4.5-Air` (bf16, 46 layers, hidden 4096, MoE 128 experts)

Control models (for “LayerNorm cosine is always high” arguments):
- `microsoft/Phi-3.5-MoE-instruct` (used in `hyunwoongko--solar-vs-glm-vs-phi/`)
- Optional extra control: `deepseek-ai/DeepSeek-V2-Lite` (used in `sionic-ai--solar-vs-glm/probe_with_control.py`)

---

## 2) Download strategy (choose one)

### A) Recommended: **Range‑based probing** (no full model download)

You do **not** download `*.safetensors` shards. Scripts:
- download small metadata files (`config.json`, `tokenizer.json`, `model.safetensors.index.json`)
- use HTTP **Range** requests to fetch only the bytes for selected tensors or sampled windows

Pros: CPU‑only, low RAM/disk, fast iteration.
Cons: requires network + HF access; more HTTP calls; depends on HF supporting Range.

### B) Offline: full shard download (only if you must)

If you need totally offline reproducibility:
- download all `*.safetensors` shards for both models
- do **not** load full models into RAM; instead read tensors directly from shards with `safetensors` / memory mapping

Rule of thumb disk:
- Solar bf16 ~ `100B params × 2 bytes ≈ 200GB` (order‑of‑magnitude)
- GLM bf16 similar scale
- Plan for **~500GB free** (both models + cache + outputs)

Avoid `transformers` full loads unless you have serious hardware/offloading.

---

## 3) Hardware expectations

### Range‑based probing (recommended)
- CPU: OK
- RAM: **8–16GB** is usually enough for the included probes; plan for **~32GB** if you build full layer×layer matrices and cache many tensors/headers.
- Disk: typically **< 2GB** (HF cache + outputs)
- Network: expect **tens to a few hundred MB** for the basic probes; more if you add many large tensor samples

### Full model load via `transformers` (not recommended)
- Solar/GLM are ~200GB bf16 each; loading whole weights naively implies **hundreds of GB RAM/VRAM**.
- You *can* sometimes use `accelerate` offload, but it’s slow and still downloads everything.

---

## 4) Environment setup

1. Create a venv (pick one):
   - `python -m venv .venv && source .venv/bin/activate`
   - `uv venv && source .venv/bin/activate`

2. Install deps for range workflow:
   - `pip install huggingface_hub requests numpy matplotlib scipy pandas`
   - `scipy` is optional but enables Hungarian alignment in some scripts.
   - If you do the offline (downloaded shards) path: `pip install safetensors`.

3. HF access:
   - Accept model terms on Hugging Face for each gated repo.
   - `huggingface-cli login` **or** set `HF_TOKEN=...`.
   - Use a cache dir with enough space: `export HF_HOME=$PWD/.hf_cache`

Security note:
- Prefer the range‑based scripts; they do **not** execute model code.
- Avoid `trust_remote_code=True` unless you’re comfortable running arbitrary code from model repos.

---

## 5) Step‑by‑step experiments

Create a per‑run output folder:
- `export OUTDIR="runs/$(date +%Y%m%d_%H%M%S)"`
- `mkdir -p "$OUTDIR"`
- `mkdir -p "$OUTDIR/artifacts" "$OUTDIR/results"`

### Experiment 0a — Pin HF revisions + file manifests (repro hygiene)

Goal: make every run reproducible by recording the exact HF commit SHA and file listing (names/sizes/LFS hashes if available).

Run (requires HF access):
```bash
python - <<'PY' > "$OUTDIR/artifacts/manifest.json"
from huggingface_hub import HfApi
import json

api = HfApi()
repos = ["upstage/Solar-Open-100B", "zai-org/GLM-4.5-Air"]
out = {}
for repo in repos:
    info = api.model_info(repo)
    out[repo] = {
        "sha": getattr(info, "sha", None),
        "siblings": [
            {
                "rfilename": s.rfilename,
                "size": getattr(s, "size", None),
                "lfs": getattr(s, "lfs", None),
            }
            for s in getattr(info, "siblings", [])
        ],
    }
print(json.dumps(out, indent=2))
PY
```

Notes:
- If either repo is gated, you may need `HF_TOKEN=...` or `huggingface-cli login` first.
- Prefer running the rest of the plan against pinned SHAs (instead of `main`) once you have them.
  - Many scripts accept a revision flag (e.g. `probe_solar_vs_glm45_air.py --revision ...`, `compare_embeddings.py --rev_solar/--rev_glm ...`); some may need small edits to stop hardcoding `main`.

### Experiment 0b — (Optional) Build a tensor inventory (headers only)

Goal: enumerate every tensor key + dtype + shape + shard (+ byte offsets if available) so we can:
- know exactly what is shape‑comparable
- run systematic exact‑match and similarity scans without hand‑curating tensor names

Approach:
- For each shard listed in `model.safetensors.index.json`, fetch and parse the safetensors header (no full tensor downloads).
- Save a table (CSV/Parquet) under `"$OUTDIR/results/"`.

This requires “our own harness” code (see section 6); the existing probe scripts are targeted, not exhaustive.

### Experiment 0 — Sanity: config + tokenizer diff

Run:
- `python sionic-ai--solar-vs-glm/probe_solar_vs_glm45_air.py --outdir "$OUTDIR/00_probe_basic"`

Outputs:
- `00_config_compare.json`
- `01_tokenizer_compare.json`

Checklist:
- Confirm both models are bf16 (`torch_dtype: bfloat16`).
- Confirm head dimension is 128 (can be inferred from attention weight shapes in other probes).
- Scan for “lineage breadcrumbs” in `00_config_compare.json`:
  - GLM‑specific fields present/absent (e.g. `num_nextn_predict_layers`)
  - closely related flags with different values (e.g. `partial_rotary_factor`, `first_k_dense_replace`)
- In `01_tokenizer_compare.json`, explicitly check tokenizer *ID* behavior (this determines whether any row‑wise embedding comparisons are meaningful):
  - `same_id_exact_matches.ratio` (are IDs preserved at all?)
  - `best_ascii_offset` + `best_offset_longest_contiguous_run.run_len` (is Solar largely a shifted/permuted GLM vocab?)
  - Special token IDs (`pad_token_id`, `bos_token_id`, `eos_token_id`) from `00_config_compare.json`
- Record the exact model revisions used (the scripts default to `main`; for strict reproducibility, rerun with a pinned `--revision`/commit hash).

### Experiment 1b — Embedding comparison for shared tokens (don’t assume stable IDs)

Goal: test whether embeddings for the **same token text** are related across models (and detect “ID‑shift/permutation” effects).

Run:
- `python sionic-ai--solar-vs-glm/compare_embeddings.py --max_tokens 500 --out_dir "$OUTDIR/01_embeddings"`

Interpretation:
- If token IDs are not preserved, comparing “row 1234 vs row 1234” is meaningless; compare by token text → this script does that.
- If many shared tokens have highly correlated embeddings (after mapping), that’s evidence of reuse/initialization; if they look random, that supports retraining / re‑indexing.

### Experiment 1 — Reproduce the headline plot: layerwise norms + router/gate

Run (same script as above):
- `python sionic-ai--solar-vs-glm/probe_solar_vs_glm45_air.py --outdir "$OUTDIR/01_layerwise"`

Key file:
- `layerwise_similarity.csv`

What to look for:
- `cosine` for norms may be high even when `pearson` is near zero → classic “cosine trap” for mostly‑positive vectors.
- `router_or_gate` similarities near 0 are expected if routers were independently trained.

Optional:
- `--align-layers` (tests whether “same layer index matches best” or if layers permute)
- `--align-experts` (perm‑invariant comparison for router/gate rows)

### Experiment 2 — Resolve the baseline dispute (the core methodological fight)

Goal: establish whether “Solar vs GLM same‑layer similarity” is **actually** exceptional compared to:
- within‑model similarities across many layer pairs
- cross‑model similarities to an unrelated control model
- metrics that remove the +1 bias (`pearson`, centered cosine)

Run:
- `python sionic-ai--solar-vs-glm/probe_final.py --outdir "$OUTDIR/02_probe_final"`

Then:
- Compare **raw cosine** vs **pearson** vs **centered cosine** (see `probe_final.py`).
- Compute a *distribution* baseline (many random layer pairs), not a single “layer 0 vs layer 10” number.
- If you extend this yourself, capture **within‑model** layer×layer matrices for both:
  - `input_layernorm.weight`
  - `post_attention_layernorm.weight`

If this still isn’t enough, write a small dedicated baseline script (recommended next code to write; see section 6).

### Experiment 3 — Strong evidence: byte‑identity tests

If any tensors are byte‑identical across the two models, that’s very strong evidence of reuse.

Start with:
- `python sionic-ai--solar-vs-glm/definitive_proof.py > "$OUTDIR/03_definitive_proof.txt"`

Then improve it (recommended):
- expand beyond “first 20 tensors”
- test multiple random contiguous byte blocks for each candidate tensor
- cover more tensor families with identical shapes (norms, router weights, biases, etc.)
- Use a tiered approach to keep bandwidth reasonable:
  - Tier 1: sample many deterministic blocks per tensor (fixed seed) and compare for exact equality (raw BF16 bytes or decoded values).
  - Tier 2: if Tier 1 finds any hits, hash the entire tensor (only for small/medium tensors; otherwise stream-hash in chunks).
- Optional “outlier fingerprint” (useful if you want a human‑interpretable hard signal):
  - For a given GLM LayerNorm vector, find the top‑K most extreme coordinates (largest `abs(x-mean)`).
  - Check whether Solar’s corresponding vector has the **same BF16 values at the same indices** across multiple layers.
  - Multiple exact coordinate/value matches on rare outliers is very hard to explain by chance.

Interpretation notes:
- **Positive match:** extremely strong evidence.
- **No matches:** not conclusive (weights could have been re‑serialized), but it raises the bar for derivation claims based only on cosine.

### Experiment 4 — Layer permutation / alignment test (does “layer index” really matter?)

For each Solar layer *L*, compare to **all** GLM layers and record the argmax.

Ways to do it:
- Use `sionic-ai--solar-vs-glm/probe_solar_vs_glm45_air.py --align-layers` (norm_pre‑based alignment)
- Or write a bespoke script that builds a full L×L similarity matrix for:
  - raw cosine
  - pearson / centered cosine
  - mean absolute difference

Quantify it (useful “Claude plan” additions):
- **Alignment rate**: fraction of Solar layers whose best match is GLM layer `L` (account for Solar’s extra +2 layers).
- **Margin**: `(best_sim - second_best_sim)` per Solar layer; small margins indicate “everything is similar”.
- Save a heatmap (e.g. `similarity_heatmap.png`) so “sharp diagonal vs diffuse cloud” is visually obvious.
- Optional significance check: shuffle GLM layer labels and recompute alignment rate/margins to estimate “by chance” expectations.

Expected outcomes:
- If Solar was initialized layer‑wise from GLM, argmax should concentrate strongly on the diagonal (after accounting for Solar’s extra +2 layers).
- If norms are all broadly similar, argmax will be diffuse and the diagonal won’t be special.

### Experiment 5 — More informative tensors than LayerNorm

LayerNorm vectors are small; routers and attention matrices are higher‑dimensional and harder to “accidentally” match.

Run the window‑sampling probe:
- `python sionic-ai--solar-vs-glm/probe_solar_glm45air.py --layers 0-45 --windows 3 --chunk_elems 262144 --try_truncation --out "$OUTDIR/05_trunc.csv"`

What this tests:
- Many projection weights (`q_proj/k_proj/v_proj/o_proj`) are shape‑compatible (or partially compatible via truncation when head counts differ).
- Whether Solar’s `q_proj` looks like a prefix/truncation of GLM’s (a stronger structural reuse signal than norms).

Optional (Gemini idea; treat as *weak* evidence on its own):
- Compare per‑layer Frobenius/L2 norms of shape‑compatible tensors (e.g. `k_proj`, `v_proj`, `mlp.gate`) and see if the “norm curve” vs layer index matches unusually well (and does *not* match controls).

### Experiment 6 — Rebuttal reproduction (Solar vs GLM vs Phi) without massive downloads

The rebuttal repo script `hyunwoongko--solar-vs-glm-vs-phi/main.py` loads full models and uses `trust_remote_code=True`.
That is usually not feasible for 100B weights on typical machines.

Preferred approach:
- re‑implement the rebuttal’s metrics (cosine + mean abs diff + within‑model matrices) using the **same Range tensor‑fetching approach** as the Sionic scripts.

If you *do* run the rebuttal script anyway:
- expect it to require extremely large resources, or `accelerate` offload configuration
- treat remote code execution risk accordingly

---

## 6) Code we likely need to write (to make this “our own testing”)

The existing scripts are a good start, but to settle disputes cleanly we should add a small, neutral harness (in this repo) that:

1. Fetches tensors either:
   - remotely via Range (HF), or
   - locally via `safetensors.safe_open` if shards are downloaded
2. Produces:
   - similarity matrices (layer×layer) for chosen tensor families
   - distribution baselines (random layer pairs; early vs late)
   - multiple metrics: cosine, centered cosine (`(x-1)`), pearson, mean abs diff, L2
   - control comparisons against Phi (and optionally another MoE)
3. Saves:
   - `revisions.json` (pin model commit hashes)
   - CSVs + plots + a short Markdown report per run

Minimal starting scripts to add:
- `scripts/fetch_tensor.py` (index→shard→header→Range fetch; plus local safetensors path mode)
- `scripts/build_tensor_inventory.py` (walk all shards, parse headers, write `{name,dtype,shape,shard,offsets,bytes}` inventory)
- `scripts/norm_baselines.py` (baseline distributions + layer alignment matrix across Solar/GLM/Phi)
- `scripts/byte_match.py` (hash random byte blocks for many tensors)
  - Include: alignment rate + per‑layer margin, plus value‑difference stats (MAD/RMSE/max/percent‑close).
  - Include: Tier‑1 sampled block matching + Tier‑2 full/stream hash escalation.
- Optional add-ons (useful, but not required to start):
  - `scripts/outlier_fingerprint.py` (find extreme coordinates in GLM vectors and test exact index/value matches in Solar)
  - `scripts/norm_dynamics.py` (per‑layer weight‑norm “curve” plots for shape‑compatible tensors)
  - `scripts/expert_permutation.py` (if/when expert tensors are shape‑compatible: Hungarian assignment to compare experts permutation‑invariantly)

---

## 7) “What would convince us?” (decision rubric)

High confidence **derived/reused**:
- repeated byte‑level matches across multiple tensors **or**
- strong diagonal layer alignment on higher‑dimensional tensors (routers, experts, attention projections where shapes allow), that does *not* appear for control models.

Likely **inconclusive / weak evidence**:
- only LayerNorm cosine is high, but centered/pearson metrics and control models show similar behavior.

High confidence **not supported by weights**:
- no meaningful alignment beyond what controls show, and no byte‑identity evidence, across a broad tensor suite.

Quick decision tree (optional shorthand):
- If within‑model norms are already ~0.9 cosine and cross‑model is also ~0.9, cosine is not discriminative → rely on centered/pearson + higher‑dim tensors.
- If argmax alignment is near‑diagonal with meaningful margins on informative tensors → evidence for derivation/initialization.
- If repeated byte‑block matches occur → very strong evidence of reuse.

---

## 8) Notes / gotchas

- Always pin model revisions (don’t rely on `main`) when publishing results.
- Don’t build statistical claims on a single baseline like “layer 0 vs layer 10”; use distributions.
- Record exactly which tensors and which slices/windows were compared.
- Keep “architecture similarity” separate from “weight reuse”; similar configs are not proof.
- Don’t redistribute weights; publish only aggregate stats, hashes, and non‑reconstructive samples.
