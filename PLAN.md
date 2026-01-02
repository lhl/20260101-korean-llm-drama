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

**Important:** LayerNorm/RMSNorm vectors are low‑information and often biased positive; raw cosine similarity can be misleading. Empirically (per community reports), deep-layer RMSNorm cosine can be ~0.99 even across *unrelated* models, so treat “high cosine on norms” as non-actionable unless paired with stronger signals (byte identity, permutation/alignment, higher‑dimensional tensors, controls, and “centered”/difference-based metrics).

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

### Experiment 0c — (Optional) Model code lineage diff (remote code)

Goal: identify obvious codebase reuse signals (copy/paste, identical helper logic, vestigial features) without running any untrusted code.

Approach:
- Download only the small text artifacts from each HF repo (no weights), e.g.:
  - `modeling_*.py`, `configuration_*.py`, `tokenization_*.py`, `generation_config.json`, `README.md`, `LICENSE`
- Run a plain-text diff and/or similarity scan.

Notes:
- This is **not** proof of weight reuse, but it can corroborate “derived implementation lineage”.
- Do **not** execute downloaded code; treat it as data.

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
- In `model.safetensors.index.json` (or your tensor inventory), scan key names for distinctive architectural artifacts:
  - attention bias tensors (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`) if present
  - next‑token prediction / MTP remnants (`nextn`, `mtp`, etc.) if present
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
- `cosine` for norms may be high even when `pearson` is near zero → classic “cosine trap” for mostly‑positive vectors (and deep-layer norms can be ~0.99 across unrelated models).
- `router_or_gate` similarities near 0 are expected if routers were independently trained.

Optional:
- `--align-layers` (tests whether “same layer index matches best” or if layers permute)
- `--align-experts` (perm‑invariant comparison for router/gate rows)

### Experiment 2 — Resolve the baseline dispute (the core methodological fight)

Goal: establish whether "Solar vs GLM same‑layer similarity" is **actually** exceptional compared to:
- within‑model similarities across many layer pairs
- cross‑model similarities to an unrelated control model
- metrics that remove the +1 bias (`pearson`, centered cosine)

Run:
- `python sionic-ai--solar-vs-glm/probe_final.py --outdir "$OUTDIR/02_probe_final"`

Then:
- Compare **raw cosine** vs **pearson** vs **centered cosine** (see `probe_final.py`).
- Compute a *distribution* baseline (many random layer pairs), not a single "layer 0 vs layer 10" number.
- If you extend this yourself, capture **within‑model** layer×layer matrices for both:
  - `input_layernorm.weight`
  - `post_attention_layernorm.weight`
- Optional sanity calibration (motivated by Reddit-style observations): compare deep-layer norm vectors against at least one unrelated model with the same hidden size (e.g. a 7B 4096-dim dense model) to quantify how often norm cosine saturates near 1.0 in practice.

**Critical methodological notes from rebuttal (2026-01-02 update):**

1. **Avoid layer 0 as baseline**: The original Sionic analysis compared layer 0 vs layers 10/20/30/40, but layer 0's `input_layernorm` is special—it directly receives embedding input and often has lower cosine similarity with other layers. Comparing layers 10/20/30 within the same model shows ~0.99 cosine similarity.

2. **Separate `input_layernorm` vs `post_attention_layernorm`**: Even at layer 0, `post_attention_layernorm` (which receives normalized input) shows high similarity (~0.92+) with other layers, while `input_layernorm` does not. Treat these tensor families separately.

3. **Centered cosine is the key discriminator**: The rebuttal shows that cross-model centered cosine drops to ~0 while within-model centered cosine remains relatively high (0.4–0.7). This strongly suggests raw cosine is misleading due to the "all positive near 1.0" distribution.

4. **Additional metrics to compute** (from rebuttal `main.py`):
   - `rel_l2`: relative L2 distance `||a-b||₂ / ||a||₂` (scale-normalized)
   - `p99_abs_diff`: 99th percentile absolute difference (tail sensitivity)
   - `cv_diff`: coefficient of variation difference `|std(a)/mean(a) - std(b)/mean(b)|`

5. **Check `k_proj` and `v_proj`**: These have matching shapes across Solar/GLM/Phi (1024×4096) and are higher-information than norms. The rebuttal found no clear derivation pattern on these either.

If this still isn't enough, write a small dedicated baseline script (recommended next code to write; see section 6).

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

**Rebuttal update (2026-01-02):** The rebuttal has been substantially expanded with:
- Centered cosine similarity analysis (key finding: cross-model drops to ~0)
- Additional metrics: `rel_l2`, `p99_abs_diff`, `cv_diff`, Pearson correlation
- Within-model layer comparisons for layers 10/20/30 (not layer 0)
- `k_proj`/`v_proj` comparisons (shape-matched at 1024×4096)
- GPT2 toy experiment (`train_gpt2.py`) demonstrating initialization bias

Preferred approach:
- re‑implement the rebuttal's metrics (cosine + centered cosine + mean abs diff + rel_l2 + within‑model matrices) using the **same Range tensor‑fetching approach** as the Sionic scripts.

If you *do* run the rebuttal script anyway:
- expect it to require extremely large resources, or `accelerate` offload configuration
- treat remote code execution risk accordingly

**Optional: replicate the GPT2 toy experiment** (`hyunwoongko--solar-vs-glm-vs-phi/train_gpt2.py`):
- This is a small-scale demonstration that ones-init LayerNorm leads to ~0.999 cross-model cosine similarity while random-init leads to ~0 similarity
- Useful pedagogical evidence that high LayerNorm cosine reflects initialization bias, not model derivation
- Can run on CPU in minutes

### Experiment 7 — Intrinsic “std‑curve” fingerprint (LLM‑Fingerprint style)

This is a different forensic angle than “are the weights close”: compare **layer‑wise parameter statistics** that tend to persist through continued training.

Source inspiration:
- `LLM-Fingerprint2/README.md` (“Intrinsic Fingerprint of LLMs…”, std patterns across attention parameters)

Goal:
- Compute a per‑layer signature like `std(W)` for selected tensor families and compare the **shape of the curve** across models.

What to compute (pick shape‑compatible tensors first):
- Attention `k_proj.weight` and `v_proj.weight` (Solar/GLM shapes match: `1024×4096`)
- MoE router/gate `mlp.gate.weight` (Solar/GLM shapes match: `128×4096`)
- Norm vectors (`input_layernorm.weight`, `post_attention_layernorm.weight`) as a low‑information reference
 - If you want to follow the paper more literally: compute attention Q/K/V/O std curves; in Solar vs GLM, Q/O are shape‑mismatched (head count), so treat Q/O as “optional/approx” (e.g., truncation to a common submatrix).

How:
- For each layer `L`, estimate:
  - `std(W_L)` (and optionally mean, skew/kurtosis, quantiles)
  - Normalize the resulting vector across layers (e.g., z‑score or min‑max) before comparing models.
- Compare Solar vs GLM with:
  - correlation (`pearsonr`) across layers
  - handle layer count mismatch (48 vs 46) by interpolating the shorter curve to the longer length (the LLM‑Fingerprint paper uses linear interpolation), or use DTW if you suspect insertions/shifts
- Always compare against at least one unrelated control model to calibrate “how similar is typical”.

Interpretation:
- High curve correlation is **supporting evidence** of lineage/recipe similarity, but not “smoking gun” on its own (many models can share training heuristics).

### Experiment 8 — Weight‑matrix fingerprint via CKA + LAP (AWM‑style)

This targets a more robust, model‑agnostic similarity metric than raw cosine on flattened tensors.

Source:
- `AWM/README.md` + `AWM/similarity_metrics.py` (CKA on attention weights + LAP alignment)

Goal:
- Compute centered kernel alignment (CKA) similarities between matched weight matrices, and optionally use LAP to align layers (and/or experts) when counts differ.

Practical adaptation for Solar vs GLM:
- AWM’s reference implementation expects **local full checkpoints** (loads full state dicts) → not realistic for 200GB‑class weights unless you’ve already downloaded them.
- Implement a *streaming* variant in our harness:
  - fetch only the tensors needed (Range or local safetensors)
  - compute **linear CKA** for matrices where shapes match (`k_proj`, `v_proj`, `mlp.gate`)
  - optionally do **LAP layer matching** using a CKA‑based cost matrix (similar to what AWM does)

Interpretation:
- Consistently high CKA on informative matrices (and diagonal‑dominant layer matching with margins) is stronger evidence than LayerNorm cosine alone.

### Experiment 9 — Black‑box / behavioral fingerprinting (optional, if you can run inference)

Weight forensics is best here, but black‑box methods are useful if:
- you only have API access, or
- you suspect **distillation** rather than direct weight reuse.

Source index:
- `Awesome-LLM-Fingerprinting/README.md` (SoK paper list; black‑box/side‑channel categories)

Possible avenues (pick based on what access you have):
- **Model equality testing / MMD** on response token sequences (requires many prompts; ideally logprobs or deterministic decoding).
- **LLMmap‑style** behavioral maps from curated prompts (requires consistent decoding settings).
- **N‑gram / stylistic classifiers** over many generations (weaker evidence; sensitive to prompting/decoding).

Practical note:
- For 100B‑class models you likely need hosted inference; bake “prompt set + decoding params + seed” into the run artifact so results are reproducible.

### Experiment 10 — Forward‑pass / representation fingerprints (optional)

If you can run forward passes (local GPU, heavy CPU offload, or a provider that exposes hidden states/logprobs), you can use forward‑pass fingerprints from the SoK list:
- intermediate activation statistics (layerwise norms/variances)
- representation encoding fingerprints (e.g., REEF‑style)
- simple linear probes on intermediate features (EasyDetector‑style)

Practical constraints for Solar/GLM:
- Running 100B‑class forward passes locally is usually unrealistic; treat this as an “if hardware/hosted access exists” add‑on.
- Tokenizers differ substantially; prefer aggregation stats (per‑layer norms over many tokens) rather than attempting token‑level alignment.

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
- `scripts/std_fingerprint.py` (LLM‑Fingerprint‑style layerwise `std(W)` curves + correlations + control baselines)
- `scripts/cka_fingerprint.py` (AWM‑style linear CKA on matched matrices + optional LAP layer alignment)
- `scripts/byte_match.py` (hash random byte blocks for many tensors)
  - Include: alignment rate + per‑layer margin, plus value‑difference stats (MAD/RMSE/max/percent‑close).
  - Include: Tier‑1 sampled block matching + Tier‑2 full/stream hash escalation.
- Optional add-ons (useful, but not required to start):
  - `scripts/outlier_fingerprint.py` (find extreme coordinates in GLM vectors and test exact index/value matches in Solar)
  - `scripts/norm_dynamics.py` (per‑layer weight‑norm “curve” plots for shape‑compatible tensors)
  - `scripts/expert_permutation.py` (if/when expert tensors are shape‑compatible: Hungarian assignment to compare experts permutation‑invariantly)
  - `scripts/behavior_fingerprint.py` (prompt set runner + output similarity metrics, if inference access exists)

---

## 7) “What would convince us?” (decision rubric)

High confidence **derived/reused**:
- repeated byte‑level matches across multiple tensors **or**
- strong diagonal layer alignment on higher‑dimensional tensors (routers, experts, attention projections where shapes allow), that does *not* appear for control models.

Likely **inconclusive / weak evidence**:
- only LayerNorm cosine is high (especially in deep layers), but centered/pearson/difference metrics and control models show similar behavior.
- only “curve similarity” (std/norm dynamics) is high without any harder signals (byte matches, alignment on informative tensors).

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
- Choose the forensic family based on access: **static weights** (strongest here) → **forward‑pass features** → **black‑box outputs** → **side‑channels** (usually out of scope unless you're auditing a deployed API).

---

## 9) Reference Repos (local copies)

These subfolders contain reference code/data and may be removed in the future:

| Folder | Source | Description |
|--------|--------|-------------|
| `sionic-ai--solar-vs-glm/` | https://github.com/sionic-ai/solar-vs-glm | Original analysis claiming Solar derived from GLM (Range-based tensor probing) |
| `hyunwoongko--solar-vs-glm-vs-phi/` | https://github.com/hyunwoongko/solar-vs-glm-vs-phi | Rebuttal comparing Solar/GLM/Phi LayerNorm similarities |
| `AWM/` | https://github.com/LUMIA-Group/AWM_Fingerprint | AWM fingerprinting method (CKA + LAP alignment) |
| `Awesome-LLM-Fingerprinting/` | https://github.com/shaoshuo-ss/Awesome-LLM-Fingerprinting | SoK paper collection on LLM fingerprinting methods |
| `LLM-Fingerprint2/` | https://github.com/HonestAGI/LLM-Fingerprint | Intrinsic fingerprint via std patterns (reupload; original disappeared) |
| `True-Story-of-Pangu/` | https://github.com/HW-whistleblower/True-Story-of-Pangu | Huawei Pangu whistleblower account (background context) |
| `distillation_detection/` | https://github.com/shqii1j/distillation_detection | NeurIPS 2025: Knowledge Distillation Detection for Open-weights Models |

### Papers (reference/)

| File | arXiv | Title |
|------|-------|-------|
| `2510.02302v1.pdf` | [2510.02302](https://arxiv.org/abs/2510.02302) | Knowledge Distillation Detection for Open-weights Models (distillation_detection) |
| `2510.06738v1.pdf` | [2510.06738](https://arxiv.org/abs/2510.06738) | AWM: Accurate Weight-Matrix Fingerprint for LLMs |
| `2502.00706v2.pdf` | [2502.00706](https://arxiv.org/abs/2502.00706) | (user-added) |

**Recommended papers to add:**
- [ ] [2507.03014](https://arxiv.org/abs/2507.03014) — Intrinsic Fingerprint of LLMs (LLM-Fingerprint2 paper)
- [ ] [2508.19843](https://arxiv.org/abs/2508.19843) — SoK: Large Language Model Copyright Auditing via Fingerprinting (Awesome-LLM-Fingerprinting)

### Discussion threads

- [r/LocalLLaMA: Upstage Solar-Open-100B public validation](https://www.reddit.com/r/LocalLLaMA/comments/1q0zst6/upstage_solaropen100b_public_validation/) — community discussion with methodological feedback incorporated into this plan

---

## Addendum A: Rebuttal repo updates (2026-01-02)

The `hyunwoongko--solar-vs-glm-vs-phi/` rebuttal received substantial updates expanding its analysis. Summary of key additions:

### New experiments and findings

1. **Within-model LayerNorm similarity (layers 10/20/30)**
   - All three models (Solar, GLM, Phi) show ~0.99 cosine similarity between different layers *within* the same model
   - This contradicts the original Sionic claim that within-model different-layer norms have low similarity

2. **Layer 0 `input_layernorm` is special**
   - The original Sionic analysis only compared layer 0 vs other layers
   - Layer 0's `input_layernorm` directly receives embedding input and behaves differently
   - `post_attention_layernorm` at layer 0 still shows high similarity (~0.92+) with other layers
   - Verified across multiple models including Qwen3-4B, Qwen3-14B

3. **Centered cosine similarity**
   - Key discriminator: subtracts mean before computing cosine
   - Cross-model centered cosine drops to **~0** (near random)
   - Within-model centered cosine remains **0.4–0.7** (meaningful structure)
   - Strongly suggests raw cosine is misleading due to "all positive near 1.0" distribution

4. **GPT2 toy experiment (`train_gpt2.py`)**
   - Trained 4 small GPT2 models with different seeds/data
   - Ones-init LayerNorm: cross-model cosine = **0.999**
   - Random-init LayerNorm: cross-model cosine = **~0**
   - Empirically demonstrates high cosine comes from initialization bias, not derivation

5. **Additional metrics**
   - Pearson correlation (equivalent to centered cosine)
   - `rel_l2`: relative L2 distance (scale-normalized)
   - `p99_abs_diff`: 99th percentile absolute difference (tail sensitivity)
   - `cv_diff`: coefficient of variation difference
   - On these metrics, Solar vs GLM is often **not** closer than Phi vs GLM

6. **`k_proj`/`v_proj` analysis**
   - Shape-matched across all three models (1024×4096)
   - No clear derivation pattern found on these higher-information tensors

### Rebuttal conclusion

> "Layernorm weight의 cosine 유사도 하나만으로 모델 파생 관계를 주장하는 것은 설득력이 약하며, 최소한 centered cosine(또는 Pearson), 절대/상대 거리 지표까지 함께 보더라도 Solar가 GLM에서 파생되었다고 볼 만한 일관된 근거는 확인되지 않았습니다."
>
> (Translation: "Claiming model derivation based solely on LayerNorm cosine similarity is unconvincing, and even when examining centered cosine/Pearson and absolute/relative distance metrics together, no consistent evidence was found that Solar was derived from GLM.")

### Implications for our plan

- **Experiment 2** updated with methodological notes from rebuttal
- **Experiment 6** updated with new rebuttal content and GPT2 toy experiment option
- Centered cosine should be treated as primary metric over raw cosine for LayerNorm comparisons
- Layer 0 should be excluded or treated separately in within-model baselines
- The higher-dimensional tensors (`k_proj`, `v_proj`, routers) remain the most informative targets
