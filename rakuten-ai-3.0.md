# RakutenAI-3.0: DeepSeek-V3 Lineage Controversy

This is a follow-up case to the Solar-Open-100B vs GLM-4.5-Air dispute: another "from scratch or derived?" argument.

Immediately after the Rakuten 3.0 release, there has been controversy due to the fact that it is architecturally a DeepSeek-V3 model, and almost certainly a post-train of DeepSeek-V3 weights, however that none of Rakuten's official statements give any attribution. There is surrounding conflation (and quite frankly Japanese racism) on Chinese hosting, misappropriation of funds, uniformed discussion of training, etc, which is no doubt *why* Rakuten sought to minimize mentioning the hot-button DeepSeek in the first place, but IMO, the complete lack of attribution or technical details has probably created much *more* controversy than what corporate/PR was trying to avoid in the first.

Below is a brief timeline and tracking of some of the evidence and analysis (of differing levels of quality) of the controversy.


## Timeline

- **2026-03-17:** Rakuten released `RakutenAI-3.0` and described it as an approximately 700B-parameter Japanese-optimized MoE model. The press release and model card do not name a base model. Instead they say it was built by "leveraging the best from the open-source community" and, in the press release, that the model was unveiled in December 2025 and is "now fine-tuned."  
  Sources: [HF model card](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/README.md), [Rakuten PR EN](https://global.rakuten.com/corp/news/press/2026/0317_01.html), [Rakuten PR JP](https://corp.rakuten.co.jp/news/press/2026/0317_01.html)

- **2026-03-19:** ITmedia asked Rakuten directly about the base model. Rakuten declined to disclose it and said the DeepSeek labeling seen on parts of Hugging Face was a site-behavior issue rather than evidence of DeepSeek lineage. On the public record, that explanation does not hold up well, because the DeepSeek reference appears not only in site metadata but in the repository's own published `config.json`.  
  Source: [ITmedia interview](https://www.itmedia.co.jp/aiplus/articles/2603/19/news099.html), [page 2](https://www.itmedia.co.jp/aiplus/articles/2603/19/news099_2.html)
  - **Source note:** ITmedia is an established Japanese online media company founded in `1999-12`, part of the SoftBank group, listed on the Tokyo Stock Exchange Prime market, and operating about 30 specialist media brands. It also publishes a reporting ethics code and a content-disclosure/corrections policy. The Rakuten article carries a named byline, `島田拓` (`Taku Shimada`). As of `2026-03-20`, ITmedia's public author archive attributes **146** articles to this byline, spanning `2024-08-28` through `2026-03-19`, mostly on AI and tech/business coverage across ITmedia AI+ and ITmedia NEWS. I did not find a fuller public staff bio beyond that archive, but the institutional and archive signals are strong enough to treat this as legitimate trade press rather than a random content farm.  
    Sources: [ITmedia author page](https://www.itmedia.co.jp/author/256965/), [author archive JSON](https://www.itmedia.co.jp/author/256965/list.json), [ITmedia company profile](https://corp.itmedia.co.jp/corp/profile/), [ITmedia media list](https://corp.itmedia.co.jp/media/), [reporting ethics code](https://corp.itmedia.co.jp/media/policy/), [content disclosure/corrections policy](https://corp.itmedia.co.jp/media/policy/guideline/)

## Rakuten Claims

### 1. Official Announcements

Rakuten's public prose does not plainly say "this is based on DeepSeek-V3." The model card and press release use softer language like "leveraging the best from the open-source community" and "based on the best models in the open-source community."

### 2. Hugging Face artifacts

The public repo artifacts point directly at DeepSeek-V3:

- The raw model card metadata [was updated](https://huggingface.co/Rakuten/RakutenAI-3.0/commit/69088b85da8ff90e8cd797d6bf5486be558b2893) to include a `DeepSeek-V3` tag.  
  Source: [Rakuten HF README raw](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/README.md)

- The public `config.json` explicitly sets:
  - `architectures: ["DeepseekV3ForCausalLM"]`
  - `model_type: "deepseek_v3"`  
  Source: [Rakuten config.json](https://huggingface.co/Rakuten/RakutenAI-3.0/raw/main/config.json)

- Rakuten's config is also a near line-by-line match to the official `deepseek-ai/DeepSeek-V3` config on the distinctive architecture fields: `num_hidden_layers: 61`, `hidden_size: 7168`, `num_attention_heads: 128`, `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`, `num_nextn_predict_layers: 1`, `vocab_size: 129280`, and the same MLA/MoE-related ranks and dimensions.  
  Source: [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)

### 3. The repo history has corrections

Since the initial public release there have been some corrections in subsequent commits. The current Hugging Face tree shows a later commit titled `Add the permission notice`, which added a `NOTICE` file after the initial upload. That `NOTICE` file contains `Copyright (c) 2023 DeepSeek` and the standard permission-notice text associated with MIT-style licensing.  
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
  Sources: [X thread by @The_AGI_WAY](https://x.com/The_AGI_WAY/status/2034463001307431217), [Thread Reader mirror](https://threadreaderapp.com/thread/2034463001307431217.html)

That is also the cleanest ethical framing here. The core criticism is not "they used an upstream open model." The core criticism is that, from an academic and transparency standpoint, the upstream base model should have been clearly credited in at least the model card.

## Independent Analyses for Weight Provenance

Below is a critical review of some weight-provenance analyses. Note: these are not comprehensive, just what's I spotted in my feed when I was looking at this.

### AWM Analysis
For testing whether `RakutenAI-3.0` uses the DeepSeek-V3 weights (and not just architecture), more analysis is required. Below I am mainly tracking some of the independent research that has popped up in my timeline. The quality varies, so each item should be judged on its own basis.

AWM ("Accurate Weight-Matrix Fingerprint for Large Language Models") is explicitly framed as a weight-based provenance method that aims to distinguish scratch-trained models from derived models even after post-training steps such as continued pretraining and fine-tuning. Source: [AWM paper](https://arxiv.org/abs/2510.06738)

A March 2026 X post by [@Aratako_LM](https://x.com/Aratako_LM/status/2034564701150195858) says an AWM-based check makes `RakutenAI-3.0` look like a post-trained descendant of DeepSeek-V3 rather than an independently scratch-trained model.

This repository has **not** independently reproduced that AWM-style result. But as public evidence, it is best read as one more weight-lineage signal pointing toward DeepSeek-V3 inheritance plus later training.

The practical public-evidence hierarchy is:

- **Clear public evidence:** DeepSeek-V3 architecture lineage.
- **Strong public indications:** DeepSeek-V3-derived weight lineage with later training.
- **Still not public:** an official plain-language acknowledgment from Rakuten.

### Shared Token Embeddings
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

#### Why the vocab-aggregated down-projection signal is also different

The down-projection result is also not the same as raw RMSNorm cosine. As Odashi describes it, the comparison is effectively probing MLP intermediate-node roles through a vocab-direction aggregate of the shared-expert down projection.

That matters because MLP/expert hidden units are permutation-sensitive: under independent random initialization, there is usually no reason to expect unit `i` in one model to preserve the same role and basis as unit `i` in another model. So a strong similarity signal there is not a trivial artifact of shared architecture in the way RMSNorm cosine can be. It points to preserved parameter structure.

#### Why the centered-cosine comment matters

One of the strongest criticisms of the original Solar/GLM evidence was that centered cosine or Pearson correlation removed the shared positive offset and made the RMSNorm signal largely disappear.

That sanity check was raised again in the Odashi discussion. The important reply was that, for the relevant vocab-direction parameters, the mean appears to be approximately zero already. If that is right, then centering should not materially change the result.

That would make this evidence qualitatively different from the old cosine trap:

- **RMSNorm case:** large shared offset, centering destroys the apparent match.
- **Embedding / vocab-aggregated MLP case:** mean appears near zero, so centering should leave most of the signal intact.

This is exactly the kind of distinction that makes one cosine-based argument weak and another potentially informative.

#### What this points to, carefully stated

These fingerprints do not specify every training step. They do not tell us exactly how much additional training was CPT versus SFT, or what intermediate internal checkpoints existed.

But they point in a much clearer direction than the old RMSNorm evidence:

- they are consistent with **inherited DeepSeek-V3 weights plus later adaptation**;
- they are hard to square with a pure scratch-training story;
- and they fit the public release framing that the March 17, 2026 checkpoint is already a post-trained ("now fine-tuned") model rather than a freshly exposed base checkpoint.

#### Was This Used In The Korean Solar/GLM Controversy?

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

### "LoRA" Claim

A March 20, 2026 note article by Hamachi, [RakutenAI-3.0とDeepSeek-V3の関係性について - ほぼLoRA適用モデル](https://note.com/hamachi_jp/n/n5ccfdedd2518), makes very confident "almost LoRA" claims and directly links its code/data basis to the GitHub repo [hama-jp/RakutenAI_Report](https://github.com/hama-jp/RakutenAI_Report). For practical purposes, our analysis treats the note article and the GitHub repo as the same authored analysis package.

For lay readers: `PEFT` means **parameter-efficient fine-tuning**. It is an umbrella term for methods that adapt a pretrained model by updating only a small subset of parameters, or by adding lightweight trainable parameters, instead of fully rewriting the whole model. `LoRA` is the most common PEFT example: it adds small low-rank adapter matrices during fine-tuning. In online discussion, people often throw around "LoRA" as shorthand for "some lightweight post-training," but that shortcut is exactly what needs to be separated here from what the public evidence actually shows.

#### What that analysis does establish

The article/repo does provide strong evidence that `RakutenAI-3.0` is derived from DeepSeek-V3:

- tokenizer identity;
- matching `config.json` / architecture;
- extremely high overall tensor similarity;
- and a pattern of small differences on top of an overwhelmingly inherited base.

That part is broadly aligned with the rest of the public evidence.

#### What the "LoRA" claim seems to be based on

The article's "LoRA" claim appears to rest on a narrower inference:

1. most weights are near-identical to DeepSeek-V3;
2. the more noticeable differences are said to concentrate in `q_a/q_b/kv_a/kv_b` low-rank attention projection tensors;
3. DeepSeek's config uses names like `q_lora_rank` and `kv_lora_rank`;
4. therefore, the author infers something like "DeepSeek-V3 + LoRA-like selective tuning."

That inference is understandable, but it is a bigger inferential leap than a lay reader might assume. In other words: it goes beyond the directly visible evidence, so it is more speculative and therefore weaker on evidentiary footing.

#### Why this is a bigger leap than it sounds

The main problem is terminological and methodological.

- DeepSeek-V3 already contains those low-rank MLA projection layers as part of the **base architecture**. They are not, by themselves, evidence of externally added LoRA adapters.
- The linked repo's current README explicitly says this: the relevant layers are **MLA low-rank projection parameters**, not external LoRA adapters.
- The repo's own `lora_parameter_analysis.py` also says the same thing in plain English: "`lora` in config naming is DeepSeek's convention for MLA ... not the LoRA fine-tuning technique."  
  Sources: [repo README](https://raw.githubusercontent.com/hama-jp/RakutenAI_Report/main/README.md), [lora_parameter_analysis.py](https://raw.githubusercontent.com/hama-jp/RakutenAI_Report/main/scripts/lora_parameter_analysis.py)

So the **lower-leap, better-supported** version of the claim is **not** "Rakuten trained this with LoRA." The better-supported claim is only that the observed changes may be concentrated in DeepSeek-V3's built-in low-rank attention projections, which is more like "LoRA-like selective tuning" than proof of standard PEFT LoRA training. The **bigger-leap** claim would be "therefore Rakuten used LoRA/PEFT as the actual training method," and that is more speculative.

#### Why the public release does not really prove PEFT LoRA

There are also reproducibility gaps in the publicly posted evidence.

- The note article claims `10,929` tensors and `61` layers, but the published CSV in the linked repo currently contains only **10 aggregate rows** for layers `0-9`, not raw per-tensor results for all 61 layers.  
  Source: [published CSV](https://raw.githubusercontent.com/hama-jp/RakutenAI_Report/main/data/comprehensive_analysis_results.csv)
- The script marketed as LoRA analysis mostly reads config values and prints theoretical low-rank parameter counts. It does not identify a training recipe.
- Even if the weight differences really do concentrate in the built-in MLA low-rank factors, that still would not by itself tell us that the actual post-training method was standard LoRA rather than some broader CPT/FT process with heavier updates in those components.

#### What the repo history suggests

The commit history of `hama-jp/RakutenAI_Report` also weakens confidence in the article's strongest framing.

- The initial public version described the result as **"LoRA implementation confirmed"** and labeled `_a_proj` / `_b_proj` weights as "LoRA parameters."  
  Source: [initial README](https://raw.githubusercontent.com/hama-jp/RakutenAI_Report/6e7dc9e/README.md)
- A later revision explicitly corrected this and stated that these are **DeepSeek-V3's built-in MLA low-rank projection layers**, not external LoRA adapters.  
  Sources: [commit `4890f85`](https://github.com/hama-jp/RakutenAI_Report/commit/4890f85), [commit `d0a09aa`](https://github.com/hama-jp/RakutenAI_Report/commit/d0a09aa)
- Shortly after that, the framing was strengthened again to **"LoRA-equivalent cost disguised as independent full model"**, which is rhetorically stronger even though it does not rest on a comparably stronger public demonstration of the actual training recipe.  
  Source: [commit `981b30f`](https://github.com/hama-jp/RakutenAI_Report/commit/981b30f)

The repo also appears to be substantively Claude-driven. The initial major commit is explicitly marked "Generated with Claude Code" and "Co-Authored-By: Claude," later commits link directly to a Claude Code session, and multiple merges come from a `hama-jp/claude/code-review-...` branch. That does not make the analysis automatically wrong, but it is relevant here because the repo both made and then corrected a basic MLA-vs-LoRA mistake while continuing to escalate the rhetorical framing. Interested readers should inspect the commit log directly rather than relying only on the latest README wording.  
Source: [commit log](https://github.com/hama-jp/RakutenAI_Report/commits/main/)

That progression does not mean the whole investigation is worthless. It does mean the repo should be read carefully as an evolving public narrative, not as a stable and methodologically settled analysis. The strongest part of the repo is still the derivation evidence. The weakest part is the leap from "differences seem concentrated in MLA low-rank factors" to "therefore this was basically LoRA."

#### Better reading

The article is useful as more public evidence for **DeepSeek-V3 derivation**. It is much less convincing as evidence that the training method itself was LoRA.

The cleaner critical reading is:

- **well supported:** DeepSeek-V3-derived model;
- **plausible:** selective tuning focused heavily on built-in MLA low-rank attention projections;
- **not well supported:** "therefore the training was LoRA."

While Rakuten has not published a full post-training recipe on top of DeepSeek-V3, a simple LoRA-only explanation looks extremely unlikely. The scale and positioning of the release, the public "now fine-tuned" framing, the broader fingerprint evidence discussed above, and the weakness of the article's own methodological basis all point away from a narrow "it was basically just LoRA" story.

## Bottom Line

The cleanest reading of the public record is:

1. Rakuten's prose avoids naming DeepSeek-V3.
2. Rakuten's published HF files nevertheless expose DeepSeek-V3 architecture lineage very directly.
3. The more important public/community fingerprints are the ones aimed at **weight lineage**: AWM-style matrix fingerprints, shared-token embeddings, and vocab-aggregated MLP/expert fingerprints.
4. Taken together, those signals point strongly to inherited DeepSeek-V3 weights with later adaptation.
5. The public evidence is therefore best read as pointing to a DeepSeek-V3-derived CPT/FT lineage, even if Rakuten's external wording remains vague.
6. The real public problem is bad framing and incomplete attribution, not the mere fact of building on an upstream open model.
7. Unlike the Solar/GLM dispute, no public AWM run or similar higher-information fingerprint seems to have been central to the earlier Korean controversy.
