> I don't read Korean. I'm interested in this back and forth.  Sionic claimed that Solar is copied from GLM and not a from-scratch pre-train?  Please give me first a translation of https://github.com/sionic-ai/solar-vs-glm and then the rebuttal https://github.com/hyunwoongko/solar-vs-glm-vs-phi
> 
> After we do this, let's do a full anaysis and see whether the claims/counterclaims stand up.
> 
> The models are online as well: https://huggingface.co/zai-org/GLM-4.5-Air and https://huggingface.co/upstage/Solar-Open-100B so we can also look at the architecture (is it the same, parameter counts and all? Are there differences?)
> 
> Also, what's up w/ the Solar-Apaches License? It doesn't look like the Apache license at all??? Can you summarize? Is that even legal to call something an Apache license that super isn't? Korean companies are just weird.

## 1) Translation — sionic-ai/solar-vs-glm (English)

**Title:** “Solar-Open-100B vs GLM-4.5-Air: weight-derivation analysis”

### Final conclusion (as written in the repo)

* **Final conclusion:** *Solar-Open-100B was derived from GLM-4.5-Air*
* **Strength of evidence:** “Decisive (182 sigma)”

---

### “Decisive evidence” section (translated)

There is an image labeled “Definitive Evidence,” followed by this central comparison:

**Within-model vs cross-model baseline comparison**

| Comparison type                                     | Cosine similarity | Explanation                  |
| --------------------------------------------------- | ----------------: | ---------------------------- |
| **Within GLM** (layer 0 vs layers 10, 20, 30, 40)   |         **0.377** | Same model, different layers |
| **Within Solar** (layer 0 vs layers 10, 20, 30, 40) |         **0.376** | Same model, different layers |
| **Solar vs GLM** (same layer number)                |         **0.989** | Different models, same layer |

**Why they say this is “decisive” (translated logic):**

* If Solar and GLM were *independent* models that are “only structurally similar,” then comparing **Solar[10] vs GLM[10]** should look like the baseline (≈ 0.38).
* But they report the observation: **Solar[10] vs GLM[10] = 0.989**.
* They compute:

  * Difference: **0.612** (61.2 percentage points)
  * Effect size: **182 sigma**
  * P-value: **< 10^-1000 (essentially 0)**
* Their interpretation: LayerNorm weights “converge randomly” during training, so two independent models should not be especially similar just because the **layer index matches**. The repo claims that the pattern “Solar[10] ≈ GLM[10] but Solar[10] ≉ GLM[20]” implies that Solar layers were directly derived from the corresponding GLM layers.

---

### “Per-tensor-type analysis” section (translated)

They present a table (image “Summary Comparison”) summarizing mean cosine similarity by tensor type:

| Tensor type                 | Mean cosine | Their interpretation |
| --------------------------- | ----------: | -------------------- |
| `input_layernorm`           |   **0.949** | “Original preserved” |
| `post_attention_layernorm`  |   **0.986** | “Original preserved” |
| `k_proj` (key projection)   |   **0.001** | “Re-trained”         |
| `v_proj` (value projection) |   **0.001** | “Re-trained”         |
| `mlp.gate` (MoE router)     |   **0.004** | “Re-trained”         |
| `embed_tokens`              |   **0.002** | “Re-trained”         |

---

### “Embedding comparison (English tokens)” section (translated)

They compare **200 shared English tokens** and report:

* Mean cosine: **0.002**
* Std: **0.015**
* cos > 0.9: **0%**
* cos < 0.1: **100%**
* Baseline (random pairs within GLM): **0.009**

**Their explanation (translated):**

* Embedding similarity near 0 is said to fit a derivation scenario where Solar expands the tokenizer:

  * GLM tokenizer (151k) + ~45k new tokens = Solar tokenizer (196k)
  * Therefore the embedding layer must be retrained (new tokens + distribution shift)
  * Result: embedding cosine ~0 (“retrained”)

They then state the “key insight”:

* LayerNorm: ~0.99 (“preserved”)
* Embedding: ~0 (“retrained”)

And they call this “selective preservation pattern” decisive evidence of derivation; they argue that for a truly independent model “all weights should be ~0” similarity.

They also show a “LayerNorm by layer” plot and describe:

* Layer 0: 0.127 (tokenizer expansion impact)
* Layer 5+: 0.95+
* Layer 40+: 0.99+

---

### “Architecture comparison” section (translated)

They provide a table stating these differences:

* hidden_size: 4096 vs 4096 (same)
* num_hidden_layers: 46 vs 48 (+2)
* num_attention_heads: 96 vs 64 (-32)
* num_key_value_heads: 8 vs 8 (same)
* n_routed_experts: 128 vs 128 (same)
* num_experts_per_tok: 8 vs 8 (same)
* n_shared_experts: 1 vs 1 (same)
* vocab_size: 151,552 vs 196,608 (+45,056)
* rope_theta: 1,000,000 vs 1,000,000 (same)
* max_position_embeddings: 131,072 vs 131,072 (same)
* num_nextn_predict_layers: 1 vs None (“removed”)

---

### “Claim 2” section (translated)

**Claim 2:** “Solar inherits almost all of GLM’s design.”
They emphasize that core design parameters match (hidden size, MoE expert count, top‑k routing, context length, RoPE theta), and argue it’s hard to view this as coincidence; they suggest Solar is likely a derived model built by modifying GLM.

---

### “Claim 3” section (translated)

**Claim 3:** “The structural evidence matches a derivation scenario.”
They list the observed pattern:

1. LayerNorm weights nearly identical
2. Embedding weights completely different
3. Attention projections completely different

They propose a possible derivation pathway:

1. Obtain GLM-4.5-Air weights
2. Expand vocab / reduce heads / add layers
3. Retrain embedding + attention + routing-related parameters
4. Keep LayerNorm weights unchanged

They conclude (translated):

> Solar-Open-100B is not “from scratch,” but a derived model based on GLM-4.5-Air after modification and retraining.

---

### Methodology (translated)

* Don’t download full safetensors; instead use HTTP Range requests to sample tensor slices (1k floats per tensor)
* Compute cosine similarity per tensor
* Use within-model inter-layer comparisons as a baseline
* If cross-model same-layer similarity is “excessively higher” than baseline, classify as weight inheritance

---

## 2) Translation — hyunwoongko/solar-vs-glm-vs-phi (rebuttal)

This rebuttal repo’s README is **already in English** (so no Korean translation needed). Here’s a faithful, cleaned-up rendition of what it argues:

### Final conclusion (as written)

* **Conclusion:** “Solar-Open-100B is NOT derived from GLM-4.5-Air.”
* **Evidence strength:** “DECISIVE.”

### Overview of their rebuttal

* The controversy relied on “high cosine similarity of LayerNorm weights between matching layer numbers.”
* The rebuttal claims this **misunderstands LayerNorm weights and cosine similarity**: LayerNorm/RMSNorm scale vectors are often **positive and similarly distributed**, so cosine similarity can be high even between unrelated models.

### Analysis 1 — “Baseline issues”

They argue the original baseline (~0.38) is wrong and that the true within-model baseline is high:

* GLM internal (layer 0 vs 10/20/30/40): **0.924**
* Solar internal (layer 0 vs 10/20/30/40): **0.923**
* Solar vs GLM (same layer): **0.989**

Therefore, they say **0.989 is not an outlier** relative to a baseline of ~0.92.

They also claim the “layer alignment” argument fails because **mismatched layers are also highly similar**:

* Solar[10] vs GLM[20]: 0.982
* Solar[10] vs GLM[30]: 0.976
* Solar[10] vs GLM[40]: 0.972

### Analysis 2 — “Actual value differences”

They argue cosine similarity measures direction, not exact equality, so you should check **mean absolute difference (MAD)**:

* Solar vs GLM MAD: **0.00923**
* GLM vs GLM (different layers) MAD: **0.00917**

They interpret this as: Solar-vs-GLM differences are comparable to within-GLM inter-layer differences, consistent with independent training.

They also tabulate other tensors and show non-LayerNorm tensors are very different (cos ~0.001, MAD ~0.03 for q/k/v/o projections).

### Analysis 3 — “Control model (Phi-4)”

They compare Phi-4 (claimed unrelated) and say LayerNorm cosine remains high:

* Phi vs GLM: 0.915
* Phi vs Solar: 0.918

Thus they conclude LayerNorm cosine similarity is not valid proof of derivation.

---

## 3) Full analysis — do the claims/counterclaims stand up?

### A) Architecture: same model or not?

From the **official Hugging Face configs**, they are clearly **similar “family” designs** (MoE, long context, similar MoE hyperparameters), but **not identical**:

**GLM-4.5-Air config highlights**

* `hidden_size`: 4096
* `num_hidden_layers`: 46
* `num_attention_heads`: 96
* `num_key_value_heads`: 8
* `n_routed_experts`: 128
* `n_shared_experts`: 1
* `num_experts_per_tok`: 8
* `max_position_embeddings`: 131072
* `rope_theta`: 1000000
* `vocab_size`: 151552
* plus GLM-specific fields like `partial_rotary_factor`, `norm_topk_prob`, `first_k_dense_replace`, `num_nextn_predict_layers`

**Solar-Open-100B config highlights**

* `hidden_size`: 4096
* `num_hidden_layers`: 48
* `num_attention_heads`: 64
* `num_key_value_heads`: 8
* `n_routed_experts`: 128
* `n_shared_experts`: 1
* `num_experts_per_tok`: 8
* `max_position_embeddings`: 131072
* `rope_theta`: 1000000
* `vocab_size`: 196608

So: **not the same architecture**, but they share a striking number of big-ticket hyperparameters.

**Parameter counts**

* GLM’s model card states GLM‑4.5‑Air is **106B total / 12B active**.
* Solar’s model card states Solar Open is **102.6B total / 12B active**, with **129 experts (top‑8 among 128 routed + 1 shared)**, and claims it was “trained entirely from scratch.”

These “12B active” similarities are consistent with both being MoE models designed for similar inference cost, not necessarily with copying.

**What architecture similarity can and cannot prove**

* It **can** support “convergent engineering” (many teams independently pick similar scaling recipes for MoE long-context models).
* It **cannot** by itself prove weight copying or derivation.

---

### B) The core disputed evidence: “LayerNorm cosine similarity proves derivation”

This dispute hinges on whether cosine similarity of LayerNorm/RMSNorm weight vectors is a reliable fingerprint.

**Sionic’s key argument**

* Within-model different-layer cosine ≈ 0.376–0.377.
* Cross-model same-layer cosine ≈ 0.989.
* Therefore the layer index alignment is “statistically impossible” unless copied (182 sigma).

**Rebuttal’s key counter**

* Within-model different-layer cosine is actually ≈ 0.923–0.924.
* Cross-model same-layer 0.989 is not an outlier; also mismatched layers remain very high.
* Therefore the “182 sigma” result is based on a wrong baseline and is invalid.

#### My assessment (based on what’s shown, without re-running code)

* The rebuttal’s critique is **methodologically plausible**: LayerNorm/RMSNorm scale parameters often cluster around similar positive values, and cosine similarity is known to be insensitive to absolute offsets/magnitude. In that setting, **cosine can be high almost everywhere**, so you need stronger tests than a single summary cosine.
* The biggest red flag for the original “182 sigma” claim is that the **baseline differs by a factor of ~2.5** between the two reports (0.377 vs 0.924). If the baseline is wrong, the whole sigma/p‑value narrative collapses.
* The rebuttal also attacks the “only matching layer numbers align” story by showing **mismatched layers also have high cosine**.  That directly targets the most “copy-like” pattern claimed.

Given only the presented materials, the “LayerNorm cosine proves derivation” claim does **not** look robust.

---

### C) “Selective preservation” (LayerNorm similar, projections/embeddings different): does that prove copying?

Sionic frames a pattern:

* LayerNorm cos ~0.99 (preserved)
* Embeddings cos ~0 (retrained)
* Attention projections cos ~0 (retrained)
  …and says this selective pattern is “decisive” for derivation.

The rebuttal responds:

* Yes, LayerNorm cosine can be high, but that’s expected and not evidence of copying.
* If something were *copied*, you should see near-zero *value differences*, not just high cosine; they show LayerNorm MAD ~0.009 and argue it’s comparable to within-model differences.

My take:

* **If only LayerNorm were “copied,”** it would be a strange “copying strategy” because LayerNorm weights carry little model knowledge compared to attention/MLP weights. You’d expect copying to target the big matrices if you were trying to shortcut training.
* Also, **Solar and GLM have different attention head counts and vocab sizes** (so several big weight matrices don’t even match in shape), which makes “direct copying of most weights” mechanically hard without some nontrivial transformation.
* The rebuttal’s suggestion—LayerNorm similarity is a weak signal—feels more consistent with how these parameters behave.

---

### D) What would actually settle “copied weights” vs “independent training”?

If you want an analysis that stands up to scrutiny, here are tests that are much harder to hand-wave away than LayerNorm cosine:

1. **Bit-level identity checks on matching-shape tensors**

   * For tensors with the exact same shape (e.g., LayerNorm vectors), sample multiple random contiguous byte blocks from the safetensors and check for exact equality.
   * If two models share copied tensors, you may find **exact byte matches** in many places (unless re-serialized differently, but safetensors is deterministic enough that chunks often match if truly identical).

2. **Elementwise difference distributions on LayerNorm**

   * Plot histogram / quantiles of `(Solar_norm - GLM_norm)` for several layers.
   * If copied then lightly fine-tuned, differences might be small but show a consistent drift pattern.
   * If independent, differences should look like two independent learned scale vectors (often still “close,” but not layer-index uniquely aligned).

3. **Try “layer permutation” tests**

   * If “layer index alignment” is the key claim, then compute similarity of Solar layer L to **all** GLM layers and see if the argmax is consistently at the same index L.
   * The rebuttal claims it’s not; the original claim implies it is.

4. **Compare a more informative tensor family**

   * MoE router/gate weights (where shapes match) could be more distinctive than LayerNorm.
   * Expert FFN weights (if shapes match) are far higher-dimensional and a stronger fingerprint than 4096-dim norm vectors.

Right now, neither repo (as presented in their READMEs) delivers this kind of “hard proof” that’s resistant to metric objections.

---

### E) Current best conclusion (with appropriate caution)

* **Architecture:** clearly similar, but not identical. This supports “same general design space,” not direct copying.
* **Weight-derivation proof:** the flagship “182 sigma” claim rests on a baseline that the rebuttal says is wrong; if the rebuttal baseline is correct, the original statistical argument fails.
* **Therefore:** based on what we can verify here, the claims *do not convincingly establish* that Solar is a weight-copied / non-from-scratch pretrain of GLM. They do establish that Solar and GLM are architecturally and design-wise close, which is interesting but not dispositive.

(Upstage also explicitly claims Solar Open was trained “entirely from scratch” in the model card, so the public-position is the opposite of the copying claim. )

---

## 4) What’s up with the “Solar-Apache License 2.0”?

### What it is

Upstage’s LICENSE file is titled **“Solar-Apache License, Version 2.0”** and explicitly says it was established by Upstage and is **not affiliated with the Apache Software Foundation**.

### Why it “doesn’t look like Apache”

Because it’s not the OSI/ASF **Apache License 2.0**. It appears to be Apache-2.0-inspired text **plus extra restrictions**, including:

* **Strategic competitor restrictions**: you may not use the Solar AI model (or derivative AI models) “directly or indirectly… by or for the benefit of a Strategic Competitor,” and you may not make it available to a Strategic Competitor.
* **Branding requirement for derivative AI models**: a derivative AI model’s name must begin with **“Built with Solar”**.

These kinds of field-of-use / competitor restrictions are **not present** in the real Apache-2.0 license.

For reference, the official Apache License 2.0 is published by ASF and has the SPDX identifier “Apache-2.0.”

### Is it “legal” to call it that?

I’m not your lawyer, but here’s the practical reality:

* Upstage is not calling it exactly “Apache License 2.0”; they’re calling it **Solar-Apache License** and they include a **non-affiliation disclaimer**.
* However, using “Apache” in the name could still be **confusing**, and depending on trademark and consumer-confusion standards, it could be risky (that’s a legal question, jurisdiction-dependent). The safe operational assumption is: **treat it as a custom, non-standard license**.

### Practical implications for developers

* If you need a genuinely permissive OSI-approved license like **Apache-2.0**, Solar-Apache is **not equivalent**.
* If you build on Solar-Open-100B, you must comply with Solar-Apache’s extra obligations (e.g., the “Built with Solar” naming requirement, competitor limitations).

And just to de-stereotype a bit: this “custom open-weight license that borrows open-source branding but adds restrictions” isn’t unique to any one country—lots of orgs do it for models.

---

If you want, next we can go one level deeper and I’ll lay out a concrete “reproducible experiment plan” that would *actually* settle the weight-derivation question (what tensors to compare, how many samples, what thresholds, and how to avoid the cosine-similarity pitfalls highlighted in the rebuttal).
