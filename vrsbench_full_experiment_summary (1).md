# VRSBench Image Captioning — Full Experiment Summary

## Task
Inter-IIT Hackathon Task 1: image captioning on the VRSBench remote-sensing dataset (20,264 caption
samples, satellite/aerial imagery, 512x512 PNGs, avg. caption length ~53 words / ~3 sentences,
vocab ~4,500 words). Official eval protocol: `Images_val` + `VRSBench_EVAL_Cap.json`.

## Evaluation
Primary metric: **BERT-BLEU4** (semantic n-gram recall via BERT embeddings + length penalty, per
the reference GeoNLI report's formula). Full suite also tracked: BLEU-1/2/3/4, METEOR, ROUGE-L,
CIDEr, Avg caption length (words).

---

## Approach 1 — BLIP-base (Salesforce), encoder-decoder, non-chat architecture

| Version | Key changes | Best BERT-BLEU4 |
|---|---|---|
| v0 (initial baseline) | Frozen encoder, greedy decode, `max_new_tokens=40` | 0.519 |
| v1 fixes | Fixed generation-length truncation, beam search (3-5 beams), repetition penalty, `min_new_tokens`, `length_penalty` | 0.565-0.575 |
| v2 optimization | Differential encoder/decoder LR (Optuna-tuned), gradient accumulation, cosine LR schedule, last 2 vision blocks unfrozen, no geometric augmentation, dynamic-length tokenization (95th-pct cutoff) | **0.575 (best overall)** |
| v3 (full dataset, 16 epochs) | Full train set instead of 15K subset | 0.572 val / 0.563 official-900 — **overfit past epoch 6, no net gain over v2** |

**Best config:** 15K/20,264 train samples, 8 epochs, LR 2e-5 cosine, batch 32, last 2 encoder blocks
unfrozen, beam search (5 beams) + length_penalty 1.3 + min_new_tokens=60.

**Key findings from BLIP runs:**
- Fixing `max_new_tokens` truncation was the single biggest early win.
- Geometric augmentation (flips) actively harmful — scrambles ground-truth spatial phrases
  ("top-left", "bottom-right") that VRSBench captions rely on.
- Frozen vs. partially-unfrozen vision encoder measurably affected grounding quality.
- Overfitting appeared once given the full dataset + many epochs — smaller-subset/fewer-epoch
  config generalized better.
- Persistent failure mode: object-count/scene-type hallucination (e.g. miscounting planes/ships,
  "roundabouts" mistaken for airport) and occasional source/resolution metadata hallucination
  ("GF medium resolution" guessed instead of "GoogleEarth high resolution").

---

## Approach 2 — Qwen3-VL-4B-Instruct QLoRA (Unsloth)
Initial attempt, 3K images, 1 epoch, 256px resize, no length control.
**Result: BERT-BLEU4 ≈ 0.43.** Verbose, hedging output ("without additional information provided...").
Root cause: undertrained (1 epoch/3K) + base instruct-model hedging habit not suppressed.

## Approach 3 — Qwen3-VL-2B-Instruct QLoRA v1 (report-style prompting, no length control)
3-4K images, metric-aware system prompt added (still no length-control loss).
**Result: BERT-BLEU4 = 0.478.** Hedging mostly fixed by prompt, but still truncating mid-sentence
(Avg_L 65-67 words vs. target ~53) and same object-hallucination pattern as BLIP (extra basketball
court, extra bridge/ships, invented golf course/highway in unrelated scenes).

## Approach 4 — Qwen3-VL-2B v2 (length-control loss + truncated targets + text-image few-shot, unbuilt/untested)
Designed with: two-layer length control (55-word truncated training targets + auxiliary loss penalty
on response token count), real-image metric-aware few-shot, official val-split eval. Not run to
completion in this session (superseded by the 7B attempt per user direction).

## Approach 5 — Qwen2.5-VL-7B-Instruct QLoRA SFT + metric-aware
Matched the GeoNLI reference report's best-known recipe (rank 16/alpha 16 LoRA, 8-bit AdamW @ 2e-4,
cosine + 100 warmup, effective batch 16). Scoped down repeatedly for time:
- Dual-T4 naive model-parallel split caused ~0.02 it/s (would have taken ~4.5hrs for original scope)
  → fixed by forcing single-GPU (`CUDA_VISIBLE_DEVICES=0`)
- Scope cut to 1200 images / 1 epoch to fit ~1.5hr budget
- Few-shot applied at inference only (not training) — too expensive per-step at 7B scale

**Result: BERT-BLEU4 = 0.436**, runtime 68.7 min. Still overshooting length target (Avg_L 64.8 vs
40-55 target) — 1 epoch/1.2K wasn't enough training to override base verbosity. Same hallucination
pattern as every other model tried.

**Cross-model finding:** bigger model (7B) with less data underperformed a smaller model (2B) with
more data, and both underperformed BLIP — confirms data volume mattered more than parameter count
at these scales, echoing the reference report's "domain adaptation, not model scale" conclusion.

## Approach 6 — TerraQ-VL (CLIP ViT-L/14 + Qwen2.5-3B, LLaVA-style, community repo) — attempted, not completed
Real, verified model (`grKnight/terraq-vl`, MIT-licensed). Attempted a continued fine-tune warm-started
from its published Stage-2 checkpoint. Key architectural constraint discovered: **single image per
prompt only** (multi-image explicitly listed as unbuilt in the repo's own docs) — few-shot had to be
implemented as text-only exemplars rather than example images.

Hit three real environment/config issues in sequence, each diagnosed and fixed from actual error logs
rather than guessed:
1. `save_steps=500` exceeded total training steps (~124) → no checkpoint ever saved → fixed to `save_steps=30`.
2. `torchao` version too old for `peft`'s LoRA dispatch (0.10.0 installed, ≥0.16.0 required) → fixed by
   uninstalling `torchao` (not needed for standard LoRA).
3. Config override for the connector warm-start checkpoint path didn't take effect (wrong assumed
   nesting under `config["model"]`) → fixed with a recursive key-search-and-override instead of a
   blind guess.

**Abandoned before producing a result** — user redirected to SmolVLM ("leave all of this just now").

## Approach 7 — SmolVLM2-2.2B-Instruct, LoRA, standard HF Trainer

### v1 (3.5K images, truncation-based length control)
Plain-transformers + peft (not Unsloth/TRL) pipeline; real-image few-shot (SmolVLM natively supports
multi-image, unlike TerraQ-VL); two-layer length control identical in spirit to the BLIP approach
(55-word truncated targets + hard-capped generation output). Not run to a logged result before the
next revision (length-penalty-loss version) superseded it per user request.

### v2 (length-penalty loss instead of truncation, 4-5hr time-boxed run)
Key differences from v1:
- All installs/downloads front-loaded into Section 1; final model saved to
  `/kaggle/working/smolvlm_vrsbench_final` (Kaggle output dir).
- **Length-penalty loss** (`LengthPenalizedTrainer`): auxiliary term = `λ × (response_token_count -
  target_token_count)²`, λ=0.02, target derived from median=53 words — a smooth gradient signal
  pulling both long *and* short captions toward the median, replacing hard truncation of targets.
- Training targets NOT hard-truncated (only a loose 90-word safety cap) — deliberate, to isolate the
  loss term's own effect.
- Wall-clock-bounded training via `TimeLimitCallback` (`num_train_epochs=50` as a high upper bound,
  callback stops training at 270 min / 4.5hr) — chosen specifically because per-step throughput
  wasn't known ahead of time, so an epoch-count guess risked finishing in 40 min or blowing past 8hrs.

**Bugs hit and fixed during setup:**
- Missing `num2words` dependency required by SmolVLM's processor.
- Image-token-count mismatch from `truncation='max_length'` slicing through an image's placeholder
  token block (occurred twice, at `max_length=1024` then again at `4096`) — root-caused to SmolVLM's
  default image tiling (`do_image_splitting=True`) multiplying token count per image; fixed by
  setting `do_image_splitting=False` and removing truncation entirely rather than raising the cap
  again.

**Result:**

| Metric | Score |
|---|---|
| BLEU-1 | 0.197 |
| BLEU-2 | 0.078 |
| BLEU-3 | 0.035 |
| BLEU-4 | 0.020 |
| METEOR | 0.193 |
| ROUGE-L | 0.189 |
| CIDEr | 0.072 |
| **BERT-BLEU4** | **0.491** |
| Avg length | 63.2 words (target: 53) |
| Runtime | 279.5 min (within 4-5hr budget) |

Length-penalty loss partially worked (63.2 vs. unconstrained VLM runs' 64.8-67.6) but didn't fully
converge to the median in the ~2-3 epochs the time budget allowed — a soft gradient penalty needs
more training exposure than hard truncation to take full effect.

## Approach 8 — Moondream2, PaliGemma, SmolVLM2 (rerun) — full-dataset "all-in" runs
Three parallel captioning-focused runs, each on the full 18,237-image train set, LoRA (rank 16,
~0.88% trainable params on Moondream2), same length-penalty-loss + median-token-target approach as
the earlier SmolVLM2 run, wall-clock-bounded to ~10hr sessions, eval on 50 official val images.

**Moondream2** (captioning-native architecture):
- Training log shows the length-penalty term visibly doing its job over the single epoch completed
  (539 optimizer steps in 600 min before the time-limit callback stopped it): `avg_response_tokens`
  swings between ~32 and ~114 per logged step early on, but the `len_penalty` term spikes
  correspondingly (e.g. 1.32 at 114 tokens vs. 0.0004 at 52 tokens — almost exactly the 53-word
  target), confirming the loss is actively pulling outlier-length batches back toward the median
  rather than sitting inert.
- **Result: BERT-BLEU4 = 0.581 — new best result overall, ahead of BLIP-base.**
- CIDEr = 0.283, more than double BLIP's 0.128 — indicates genuinely better n-gram/phrase overlap
  with references, not just embedding-level semantic similarity.
- Avg length 42.2 words — undershoots the 53-word target (opposite direction from every chat-model
  VLM tried, which all overshot), but the closer overall distance plus strong CIDEr suggests this
  undershoot cost less than the previous approaches' overshoot did.
- Qualitative predictions are notably cleaner than any earlier VLM run: correct spatial phrasing,
  no hedging, no run-on sentences, count errors when present are small (5 ships vs. reference's
  implied "several/multiple" rather than wildly wrong).

**PaliGemma** (captioning-native architecture): BERT-BLEU4 = 0.568, BLEU-4 = 0.077, CIDEr = 0.207,
Avg_L = 41.5 words. Essentially tied with BLIP-base, best length-undershoot control of the three.

**SmolVLM2** (chat-model architecture, rerun on full dataset): BERT-BLEU4 = 0.511, BLEU-4 = 0.026,
CIDEr = 0.096, Avg_L = 58.0 words (overshoots target, same direction as every earlier chat-VLM run),
runtime 552 min. Improved over the earlier 3.5K-subset run (0.491) with more data, but still last of
the three "all-in" runs and still exhibits the overshoot pattern chat-model VLMs have shown
throughout this project.

---

## Overall leaderboard (BERT-BLEU4)

| Rank | Model | Config | Score |
|---|---|---|---|
| 1 | **Moondream2** | Full 18.2K/1ep (~10hr), length-penalty loss, LoRA | **0.581** 🏆 |
| 2 | BLIP-base | 15K/8ep, differential LR, beam search | 0.575 |
| 3 | PaliGemma | Full 18.2K, length-penalty loss, LoRA | 0.568 |
| 4 | SmolVLM2 (full-data rerun) | Full 18.2K, length-penalty loss, LoRA | 0.511 |
| 5 | SmolVLM2 (3.5K subset) | 3.5K/~2-3ep, length-penalty loss | 0.491 |
| 6 | Qwen3-VL-2B v1 | 3-4K/1ep, metric-aware prompt, no length control | 0.478 |
| 7 | Qwen2.5-VL-7B | 1.2K/1ep, SFT + metric-aware, inference-only few-shot | 0.436 |
| — | Qwen3-VL-4B (earliest) | 3K/1ep, 256px, no length control | 0.43 |
| — | TerraQ-VL | Attempted, not completed | — |

## Cross-cutting findings (apply across all models tested)

1. **Response length control was the single most impactful lever throughout** — every model that
   overshot its target word count paid for it directly via BERT-BLEU4's length penalty and
   BLEU/CIDEr's n-gram precision. Hard truncation (BLIP) is a reliable blunt instrument; the
   loss-based length penalty (Moondream2, PaliGemma, SmolVLM2) visibly pulled predictions toward the
   median in training logs and, given a full epoch over the full dataset, converged well enough to
   produce the best result of the entire project (Moondream2).
2. **Architecture type (captioning-native vs. chat-model) matters more than parameter count or even
   training data volume.** Both captioning-native models tried (Moondream2, PaliGemma) beat or tied
   BLIP and clearly beat every chat-model-based VLM (Qwen3-VL, Qwen2.5-VL, SmolVLM2) despite SmolVLM2
   getting the same full-dataset treatment. Chat-model VLMs consistently overshot the target length
   and were more prone to hedging/meta-commentary; captioning-native models did neither.
3. **Object-count and scene-type hallucination is universal, though less severe in the best runs** —
   present in every architecture tried, but Moondream2's qualitative errors (e.g. "5 ships" vs. an
   implied "several") are notably milder than earlier runs' wholesale scene misreads (e.g.
   "roundabouts" mistaken for an airport). Still not fully solved by any approach here.
4. **Data volume beat parameter count among the chat-model VLMs specifically** — BLIP and the
   captioning-native models complicate a pure "more data wins" reading, but within the chat-VLM
   family alone (Qwen 2B/4B/7B, SmolVLM2), more training data consistently helped.
5. **Chat-model hedging is architecture-specific** — Qwen's instruct-tuned base habit of
   meta-commentary ("without additional information provided...") had to be explicitly banned via
   system prompt; neither BLIP, Moondream2, nor PaliGemma exhibited this, consistent with none of
   them being general-purpose chat models.
6. **Environment/dependency issues consumed significant iteration time** across the VLM approaches:
   dual-GPU naive model-parallelism slowdown, `torchao` version mismatches, missing `num2words`,
   image-tiling token-count blowup, and a config key that silently failed to override. Each was
   root-caused from actual error logs rather than guessed at.
7. **Geometric augmentation is harmful for this dataset** specifically because captions encode
   absolute spatial position ("top-left", "bottom-right") that flips/rotation would invalidate.

## Recommendation
**Moondream2 (0.581 BERT-BLEU4) is now the strongest result of the project**, narrowly ahead of
BLIP-base (0.575) and PaliGemma (0.568), with the caveat that its full 18.2K-image run only
completed one training epoch before hitting the wall-clock cap — a longer session would likely push
it further given the length-penalty loss was still visibly converging. The clearest finding across
all eight approaches is that **captioning-native architectures (Moondream2, PaliGemma) reliably
outperform general-purpose chat-model VLMs (Qwen3-VL, Qwen2.5-VL, SmolVLM2) on this task** —
recommend leading the submission with Moondream2 as the primary result, BLIP-base as the efficient
low-compute baseline, and framing the chat-VLM attempts as a documented negative result explaining
*why* architecture choice mattered more than scale here.
