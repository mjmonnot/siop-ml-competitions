# Known Landmines

Eight traps that cost competition teams real score points in 2024, or that I hit (or avoided) while building this post-hoc reconstruction. They're ordered roughly by how much score each one is worth.

Format borrowed from the 2026 meta-analysis repo: each landmine is a concrete failure mode, what it costs you, how to detect it, and how to avoid it.

---

## Landmine 1 — Using an LLM for calibrated regression

**What it costs.** ~4 points of Pearson correlation on the clarity task. PAID Team's deck explicitly says "Tried GPT4 (failed). Tried fine-tuned GPT3.5 (failed)" before they switched to a fine-tuned DeBERTa-V3-base and scored .816. The teams that stayed with LLM-only approaches on clarity capped at around .74.

**Why it happens.** Chat models are trained to produce plausible text, not calibrated continuous values. Ask GPT-4o "rate the clarity of this item from 1.0 to 7.0" and you'll get answers that cluster around 5.0 with too little variance. Pearson correlation cares about ordering AND spread; the LLM gives you decent ordering and lousy spread.

**How to detect it.** Run the LLM clarity pipeline on the dev set, compute Pearson r, and look at the standard deviation of your predictions vs. the standard deviation of the ground truth. If the prediction std is materially smaller (e.g., 0.7 vs. 1.4), you're under-spread.

**How to avoid it.** Fine-tune a regression model. DeBERTa-V3-base with MSE loss converges in under 20 minutes on a single GPU on the 2024 clarity training set. The alternative — wrapping the LLM with feature engineering (Hungry Llama's approach) — helps but doesn't fully fix the calibration issue, and you've now built a more complex system than the fine-tune.

The clarity adapter in this repo is intentionally LLM-only because it's pedagogical: the notebook walks through how far you can get without fine-tuning (~.70). Don't ship it.

---

## Landmine 2 — Using "all candidates" few-shot for the interview task

**What it costs.** ~7 points of cosine similarity. PAID Team's "question-centered" interview approach (collect all responses to Q5 across train/val/test, use as few-shot) scored .440 — last of the four winners. The teams that preserved candidate-specific style scored .49–.51.

**Why it happens.** The interview metric is cosine similarity between *this candidate's* generated and actual Q5 response. If you few-shot on a hundred other candidates' Q5 responses, the model learns "here's how Q5 is typically answered" and generates a generic answer. Generic answers have decent cosine to a generic ground truth and terrible cosine to a specific one.

**How to detect it.** Compute the pairwise cosine similarity between generated and actual responses; if the variance of your similarity scores is much lower than the variance across pairs of arbitrary candidates, you've generated something too generic.

**How to avoid it.** Don't add external few-shot examples at all. The four prior responses ARE the context. If you want to do better than that, condition on Big-5 personality inferred from the prior responses (Hungry Llama's approach, +7 cosine points), or generate N candidates and pick the one most similar to the prior responses (Akben's approach, +5 cosine points).

The interview adapter in this repo follows the no-external-examples rule by default and has an opt-in Akben-style reranker via `harness.call_consistent`.

---

## Landmine 3 — Skipping self-consistency on classification tasks

**What it costs.** 1–3 points of accuracy on each of empathy and fairness when the model is near its accuracy ceiling. Akben tied PAID on fairness (.828 vs. .828) and beat PAID on empathy (.608 vs. .580) using majority-vote self-consistency at N=5.

**Why it happens.** Temperature-0 chat completions are nominally deterministic, so teams skip self-consistency thinking it's free of value. But the OpenAI seed parameter (which we use) only makes results *highly likely* to be reproducible, not perfectly so — and at non-zero temperature with diverse few-shot orderings, the model genuinely flips on edge cases.

**How to detect it.** Re-run the same dev set twice at temperature 0.7. Count the rows where you get different predictions. If it's more than a couple of percent of the test set, self-consistency will help.

**How to avoid it.** Wrap the call in `call_consistent(spec, n=5, reduce=mode_reducer)`. The cost is 5x API calls; the gain is roughly 1–3 points of accuracy when the underlying model is uncertain. For tasks where you're already at the accuracy ceiling (e.g., a GPT-4 fairness model at .85), self-consistency may not move the needle.

The harness supports this via `--self-consistency 5` on the CLI. It's off by default so the cost matches what PAID Team actually paid.

---

## Landmine 4 — Random few-shot selection when similarity is available

**What it costs.** ~2 points of accuracy on empathy. This isn't reported by any 2024 team because none of them used similarity selection; it's a 2026-hindsight finding.

**Why it happens.** With a fixed token budget for few-shot context, using random examples wastes tokens on examples that aren't close to the test row. Empathy in particular has emails of very different lengths and tones; if your random sample doesn't include an example with a similar tone to the test row, you're under-conditioning the model.

**How to detect it.** Compute the average cosine similarity between each test row and its few-shot context. If the random-selected average is much lower than what you'd get by picking the K nearest neighbors, you're leaving signal on the table.

**How to avoid it.** Use `pick_similar` from `examples.py`. The `--similarity-examples` CLI flag is opt-in on a per-task basis.

**The caveat that makes this Landmine 4 and not Landmine 1.** Similarity-based selection hurts on clarity (see Landmine 6) and is irrelevant for interview (no external few-shot). It helps on empathy and fairness. Don't apply it blindly.

---

## Landmine 5 — Trusting LLM yes/no outputs without structured outputs

**What it costs.** 0.5–2 points of accuracy, depending on how often the model deviates from the requested response format.

**Why it happens.** The 2024 competition predates OpenAI's strict structured-output JSON mode (released August 2024). Teams either used regex on free-text outputs or accepted occasional parse failures as a fixed cost.

**How to detect it.** Log every model output. Count how often it deviates from the expected format. If you're seeing 1%+ deviations, you're losing accuracy on those rows.

**How to avoid it.** Pass `response_format={"type": "json_schema", ...}` with strict mode enabled. The adapters in this repo do this for empathy and fairness. The empathy schema enforces `{"label": 0 | 1}` and the fairness schema enforces `{"choice": "first" | "second"}`. Once you set strict=true with a recent gpt-4o, the model literally cannot return an off-format response.

The PAID Team notebooks (released April 2024) didn't use structured outputs and had defensive regex parsing in their notebooks. Adding strict JSON to their pipeline today would recover the silent parse failures for free.

---

## Landmine 6 — Similarity-based few-shot on the clarity task

**What it costs.** ~3 points of Pearson correlation. I caught this one experimentally — when I first added similarity-based few-shot to the harness, clarity dropped from r=.71 to r=.68 on the dev set.

**Why it happens.** Personality test items in the clarity training set cluster tightly in semantic space (they're all short statements about psychological dispositions). Picking the K most-similar training items as few-shot means you're showing the model K examples with very similar *content* and very similar *clarity ratings*. The model learns the local cluster's rating range and predicts inside it — even when the test item is actually unusually clear or unclear.

In other words: similarity selection makes the model's predictions more peaked, which is the *opposite* of what Pearson correlation rewards.

**How to detect it.** Compute the standard deviation of your predictions. Compare it to the standard deviation of the ground truth. Similarity-selected few-shot will compress the prediction range.

**How to avoid it.** For clarity, use random few-shot selection. The harness's CLI does this by default; the `--similarity-examples` flag has no effect on clarity. (Internally the flag is checked per-task in `run.py`.)

This is a useful reminder that "use the most similar training examples" is a heuristic for tasks where the answer should resemble the answer for similar inputs. It fails for regression tasks where you want to span the response distribution.

---

## Landmine 7 — Putting imputation inside the pipeline instead of after it

**What it costs.** Variable. The risk is high (a malformed pipeline run silently produces all-default predictions that look like a valid submission).

**Why it happens.** It's tempting to handle parse failures by substituting the train-set grand mean or the majority class inside the adapter's `parse` method. The clarity adapter in this repo does exactly that: if the regex can't find a number, it returns 5.4. The risk is that if the pipeline misroutes — say, the input file is empty or the API key is wrong — every row gets 5.4 and you ship a submission that's exactly the grand-mean predictor.

**How to detect it.** After running the pipeline, compute the standard deviation of your predictions. If it's near zero, you've shipped a constant.

**How to avoid it.** Two layers of defense:

1. The adapter's `parse` returns a default only when it genuinely cannot parse. It does NOT replace empty model outputs with defaults — empty output means an upstream failure, and you want to know about that.

2. Imputation for *missing rows* (rows the pipeline didn't process because of an error) should happen *outside* the pipeline, after writing the submission CSV. The recipe from the 2026 repo applies here too: write blanks for failures, fill blanks at submission time with whatever default you want.

This repo doesn't currently ship a separate imputation step because the harness's retry logic should handle transient failures. If you find yourself with blank rows in a submission, that's a bug worth investigating, not a default to paper over.

---

## Landmine 8 — Optimizing on dev to the point of overfitting

**What it costs.** The published score distributions in the official deck show that several teams' test-phase scores dropped meaningfully vs. their dev-phase scores. The composite score median was .60 on dev and .57 on test — a 3-point drift, which on an 11-row leaderboard is the difference between 1st and 6th.

**Why it happens.** The competition format (dev set with leaderboard, then a held-out test set) invites overfitting on dev. Teams iterate prompt wording, few-shot K, temperature, and other hyperparameters to maximize dev score, and some of that gain is dev-set noise.

**How to detect it.** Hold back a portion of the dev set as a true validation split. Iterate on the other portion. Compare. If your "validation" portion tracks the held-back portion within a point or two, you're probably not overfitting. If it doesn't, you are.

**How to avoid it.** Set a hard limit on how many dev-set evaluations you do, and lean on principled hyperparameter choices rather than empirical sweeps. Hungry Llama's deck notes "best solution up until 2 weeks prior to test data release was randomly guessing between values of 5.5 and 6.5 but using a value of 3 for cases with any phrase with 'am', a comma, or negation" — that's a rule that was clearly tuned to specific dev-set quirks and has no reason to transfer.

This repo deliberately doesn't sweep over hyperparameters. The K-of-N few-shot sizes (16 for empathy, 24 for fairness, 24 for clarity), the self-consistency N (5 when used), the interview max_words (120), and the temperature (0.0 by default) are all defended in the source code with one or two sentences each. If you fork this and start tuning these on dev to chase the next half-point, you're walking into Landmine 8.
