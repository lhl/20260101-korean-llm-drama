# RakutenAI-3.0: DeepSeek-V3 Lineage Controversy

This is a follow-up case to the Solar-Open-100B vs GLM-4.5-Air dispute: another "from scratch or derived?" argument, but with much cleaner public architecture evidence and stronger public fingerprints pointing to a DeepSeek-V3-derived post-trained model.

## Timeline

- **2026-03-17:** Rakuten released `RakutenAI-3.0` and described it as an approximately 700B-parameter Japanese-optimized MoE model. The press release and model card do not name a base model. Instead they say it was built by "leveraging the best from the open-source community" and, in the press release, that the model was unveiled in December 2025 and is "now fine-tuned."  
  Sources: [HF model card](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/README.md), [Rakuten PR EN](https://global.rakuten.com/corp/news/press/2026/0317_01.html), [Rakuten PR JP](https://corp.rakuten.co.jp/news/press/2026/0317_01.html)

- **2026-03-19:** ITmedia asked Rakuten directly about the base model. Rakuten declined to disclose it and said the DeepSeek labeling seen on parts of Hugging Face was a site-behavior issue rather than evidence of DeepSeek lineage. On the public record, that explanation does not hold up well, because the DeepSeek reference appears not only in site metadata but in the repository's own published `config.json`.  
  Source: [ITmedia interview](https://www.itmedia.co.jp/aiplus/articles/2603/19/news099.html), [page 2](https://www.itmedia.co.jp/aiplus/articles/2603/19/news099_2.html)

## What Is Publicly Visible

### 1. The prose stays vague

Rakuten's public prose does not plainly say "this is based on DeepSeek-V3." The model card and press release use softer language like "leveraging the best from the open-source community" and "based on the best models in the open-source community."

### 2. The raw Hugging Face artifacts are not vague

The public repo artifacts point directly at DeepSeek-V3:

- The raw model card metadata [was updated](https://huggingface.co/Rakuten/RakutenAI-3.0/commit/69088b85da8ff90e8cd797d6bf5486be558b2893) to include a `DeepSeek-V3` tag.  
  Source: [Rakuten HF README raw](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/README.md)

- The public `config.json` explicitly sets:
  - `architectures: ["DeepseekV3ForCausalLM"]`
  - `model_type: "deepseek_v3"`  
  Source: [Rakuten config.json](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/config.json)

- Rakuten's config is also a near line-by-line match to the official `deepseek-ai/DeepSeek-V3` config on the distinctive architecture fields: `num_hidden_layers: 61`, `hidden_size: 7168`, `num_attention_heads: 128`, `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`, `num_nextn_predict_layers: 1`, `vocab_size: 129280`, and the same MLA/MoE-related ranks and dimensions.  
  Source: [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)

The corrected tag alone would be weak evidence because HF tags can be noisy. The `config.json` is the stronger point. Taken together, this is hard to reconcile with the claim that DeepSeek appeared only because of Hugging Face auto-labeling or auto-calculation. The DeepSeek reference is embedded in the published repository files themselves.

### 3. The current repo history also shows a later DeepSeek notice

There is also a licensing/transparency wrinkle. The current Hugging Face tree shows a later commit titled `Add the permission notice`, which added a `NOTICE` file after the initial upload. That `NOTICE` file contains `Copyright (c) 2023 DeepSeek` and the standard permission-notice text associated with MIT-style licensing.  
Sources: [HF tree/history view](https://huggingface.co/Rakuten/RakutenAI-3.0/tree/main), [HF NOTICE raw](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/NOTICE)

That does not, by itself, answer every downstream licensing question, and this note is not legal advice. But as a matter of public-facing transparency, it is another strong sign that the repository maintainers were acknowledging an upstream DeepSeek component after the fact, not dealing with a purely Hugging Face-generated labeling artifact.

This wrinkle was also noticed publicly in outside commentary, including a March 19, 2026 Qiita post discussing the `config.json` and `NOTICE` contents.  
Source: [Qiita article](https://qiita.com/GeneLab_999/items/98ee926b81df797a202c)

## What The Public Record Shows

### Public evidence strongly supports architecture lineage

At minimum, `RakutenAI-3.0` is publicly exposed as a DeepSeek-V3-family architecture. That much is not in doubt once the raw `config.json` is inspected. Architecture evidence, however, is not the same thing as weight-provenance evidence. The `config.json` by itself does not settle whether the model inherited DeepSeek-V3 weights or only its architecture.

### But the broader external evidence points beyond "same architecture"

Once the public fingerprinting evidence is added, the picture is stronger than "same architecture, unknown weights." The architecture files, the shared-token embedding similarity reports, the shared-expert/MLP fingerprint reports, and the release framing that the model is "now fine-tuned" all point in the same direction: a DeepSeek-V3-derived model that was further adapted rather than an independently scratch-trained sibling.

## What This Does And Does Not Mean

There has also been obvious [overreaction](https://x.com/cgbeginner/status/2033979136608506264) around this story. A fairer summary is that the main problem is **framing and disclosure**, not that using DeepSeek-derived open weights is inherently improper or that Rakuten users' data is somehow being sent to DeepSeek servers.

Outside commentary has pushed back on three common confusions:

- a DeepSeek-derived open-weight model running on Rakuten infrastructure does **not** imply runtime data is sent to DeepSeek;
- a DeepSeek-derived model is not necessarily "just the original model unchanged" if it has undergone substantial Japanese-focused continued training and fine-tuning;
- the public issue is mainly that the base model lineage was not clearly credited in the model card and external messaging.  
  Source: [Thread Reader summary](https://threadreaderapp.com/thread/2034463001307431217.html)

That is also the cleanest ethical framing here. The core criticism is not "they used an upstream open model." The core criticism is that, from an academic and transparency standpoint, the upstream base model should have been clearly credited in the model card and related public materials.

## DeepSeek-V3 Base Model

For testing whether `RakutenAI-3.0` uses the DeepSeek-V3 weights (and not just architecture), more analysis is required. So far there have been two independent methods researchers have applied.

AWM ("Accurate Weight-Matrix Fingerprint for Large Language Models") is explicitly framed as a weight-based provenance method that aims to distinguish scratch-trained models from derived models even after post-training steps such as continued pretraining and fine-tuning. Source: [AWM paper](https://arxiv.org/abs/2510.06738)

A March 2026 X post by [@Aratako_LM](https://x.com/Aratako_LM/status/2034564701150195858) says an AWM-based check makes `RakutenAI-3.0` look like a post-trained descendant of DeepSeek-V3 rather than an independently scratch-trained model.

This repository has **not** independently reproduced that AWM-style result. But as public evidence, it is best read as one more weight-lineage signal pointing toward DeepSeek-V3 inheritance plus later training.

The practical public-evidence hierarchy is:

- **Clear public evidence:** DeepSeek-V3 architecture lineage.
- **Strong public indications:** DeepSeek-V3-derived weight lineage with later training.
- **Still not public:** an official plain-language acknowledgment from Rakuten.

## Independent Fingerprints Beyond AWM

Another useful community line of evidence comes from a March 2026 thread by [@odashi_t](https://x.com/odashi_t/status/2034557954331181528). Based on the reported results in that thread and follow-up discussion:

- DeepSeek-V3 and `RakutenAI-3.0` reportedly have very high cosine similarity for shared token embeddings.
- The lowest-similarity shared embeddings are said to cluster in punctuation and Chinese/Japanese token pieces rather than appearing uniformly random.
- Odashi also reports a signal from a vocab-direction aggregate of a shared-expert down projection, and describes that as a reasonable proxy for similarity of MLP intermediate-node roles.

This is importantly different from the earlier Solar/GLM RMSNorm fight and is better understood as weight-lineage evidence rather than architecture trivia.

### Why shared-token embeddings are stronger than RMSNorm vectors

Shared-token embeddings are anchored to specific token IDs. If two models share the same tokenizer and token IDs, row-to-row embedding comparison is comparing the parameter attached to the same token, not just some low-information normalization scale vector.

That makes the interpretation very different from RMSNorm/LayerNorm cosine:

- RMSNorm weights are small, low-information, mostly-positive scale parameters and are vulnerable to cosine inflation from shared offset/init effects.
- Token embeddings are high-information parameters tied to particular lexical items.
- If most shared token rows remain highly similar while a small subset of punctuation and Chinese/Japanese pieces drift, that looks like **selective adaptation of inherited weights**.

In other words, this is closer to a fingerprint of preserved parameter identity with localized edits.

### Why the vocab-aggregated down-projection signal is also different

The down-projection result is also not the same as raw RMSNorm cosine. As Odashi describes it, the comparison is effectively probing MLP intermediate-node roles through a vocab-direction aggregate of the shared-expert down projection.

That matters because MLP/expert hidden units are permutation-sensitive: under independent random initialization, there is usually no reason to expect unit `i` in one model to preserve the same role and basis as unit `i` in another model. So a strong similarity signal there is not a trivial artifact of shared architecture in the way RMSNorm cosine can be. It points to preserved parameter structure.

### Why the centered-cosine comment matters

One of the strongest criticisms of the original Solar/GLM evidence was that centered cosine or Pearson correlation removed the shared positive offset and made the RMSNorm signal largely disappear.

That sanity check was raised again in the Odashi discussion. The important reply was that, for the relevant vocab-direction parameters, the mean appears to be approximately zero already. If that is right, then centering should not materially change the result.

That would make this evidence qualitatively different from the old cosine trap:

- **RMSNorm case:** large shared offset, centering destroys the apparent match.
- **Embedding / vocab-aggregated MLP case:** mean appears near zero, so centering should leave most of the signal intact.

This is exactly the kind of distinction that makes one cosine-based argument weak and another potentially informative.

### What this points to, carefully stated

These fingerprints do not specify every training step. They do not tell us exactly how much additional training was CPT versus SFT, or what intermediate internal checkpoints existed.

But they point in a much clearer direction than the old RMSNorm evidence:

- they are consistent with **inherited DeepSeek-V3 weights plus later adaptation**;
- they are hard to square with a pure scratch-training story;
- and they fit the public release framing that the March 17, 2026 checkpoint is already a post-trained ("now fine-tuned") model rather than a freshly exposed base checkpoint.

So the clean public-facing claim is: these are the kinds of fingerprints that **point toward a DeepSeek-V3-derived CPT/FT lineage**, rather than merely showing "same architecture" or repeating the old RMSNorm cosine pathology.

## Was This Used In The Korean Solar/GLM Controversy?

Short answer: **not publicly, as far as I can tell**.

The January 1, 2026 Solar/GLM dispute was mostly about:

- raw LayerNorm/RMSNorm cosine similarity,
- within-model vs cross-model baselines,
- centered cosine / Pearson rebuttals,
- and general arguments about false positives from norm-vector comparisons.

The public repos in that dispute do **not** appear to contain an AWM run or an AWM-style CKA+LAP weight-matrix fingerprint. In fact, the later writeups explicitly point to stronger fingerprinting methods such as HuRef and REEF as future work rather than presenting an AWM result:

- [sionic-ai/solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)
- [hyunwoongko/solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)

So this Rakuten case should be understood as:

- the **same kind** of controversy,
- but **not** the same public methodology,
- and arguably a better fit for stronger provenance tools than the original Solar/GLM exchange.

That is also why this repository's own [PLAN.md](PLAN.md) later elevated AWM-style fingerprinting as a stronger follow-up experiment after the Solar/GLM argument over LayerNorm cosine got stuck.

## Bottom Line

The cleanest reading of the public record is:

1. Rakuten's prose avoids naming DeepSeek-V3.
2. Rakuten's published HF files nevertheless expose DeepSeek-V3 architecture lineage very directly.
3. The more important public/community fingerprints are the ones aimed at **weight lineage**: AWM-style matrix fingerprints, shared-token embeddings, and vocab-aggregated MLP/expert fingerprints.
4. Taken together, those signals point strongly to inherited DeepSeek-V3 weights with later adaptation.
5. The public evidence is therefore best read as pointing to a DeepSeek-V3-derived CPT/FT lineage, even if Rakuten's external wording remains vague.
6. The real public problem is bad framing and incomplete attribution, not the mere fact of building on an upstream open model.
7. Unlike the Solar/GLM dispute, no public AWM run or similar higher-information fingerprint seems to have been central to the earlier Korean controversy.

So yes: this is a useful follow-up to the Korean case, but it is better thought of as a **new provenance dispute with stronger architecture evidence and a more relevant fingerprinting method**, not as a rerun of the exact same public analysis.
