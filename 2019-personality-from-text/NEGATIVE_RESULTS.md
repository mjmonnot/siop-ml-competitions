# Negative / null results log

Directions tried that did **not** help (or actively hurt). As informative as the
wins. All numbers are Train-CV (OOF) / Dev unless noted.

## Representations / embedders
- **Plain SBERT/embedding + Ridge on all-text** underperforms TF-IDF, reproducing
  the brief's warning. `e5-large-v2` all-text + Ridge: OOF 0.136 / Dev 0.147 —
  far below char TF-IDF (OOF 0.215 / Dev 0.259). Short, style-laden text + a
  semantic embedder + a linear head loses the stylistic variance.
- **`bge-large-en-v1.5` and `gte-large` are much weaker than `e5-large-v2`** for
  the (winning) per-prompt-SVR recipe: bge OOF 0.224 / gte OOF 0.217 vs e5
  OOF 0.285. "Bigger / newer MTEB rank" did not translate to this task.
- **Per-prompt concatenation only helps with a nonlinear head.** Per-prompt e5 +
  *Ridge* (`embp`) was mediocre (OOF 0.171); the gain comes from per-prompt + RBF
  SVR together (OOF 0.285).

## TF-IDF tuning
- **Word bigrams/trigrams overfit.** word (1,2) and (1,3) were worse on Dev than
  word **unigrams** (1,1) (Dev 0.24 vs 0.267). Bigrams add variance, not signal,
  at n≈1k.
- **Matched-prompt-only TF-IDF models are weak** (OOF ~0.10–0.14): a single
  prompt's text is too sparse per trait. Dropped from the final stack.

## Heads
- **Ridge on raw embeddings** is weak (see above); **RBF SVR** is essential on
  embeddings (OOF 0.136 → 0.285 for the per-prompt e5 features).
- **HistGradientBoosting on engineered features** (~OOF 0.143) was no better than
  Ridge on the same features; kept only as a small diversity base.
- **PCA before SVR** (128/256 comps) *reduced* Train-CV/Dev (OOF 0.273/0.272 vs
  0.285 with no PCA), so the final engine uses full embeddings. (Open question:
  PCA denoising might transfer better to the harder Test, but this was not
  validated to preserve the single-touch rule.)

## SVR hyperparameters
- **C is nearly inert** for C ≥ 2 (predictions saturate); **gamma is the active
  knob**. gamma="scale" (≈2e-4) is optimal; gamma=5e-4 overfits (OOF 0.265),
  gamma=1e-3 collapses (OOF 0.06). C=2 marginally beats C=4 (≈+0.001).

## Meta-layer
- **Cross-trait meta-features (target-conditioning) overfit.** Giving each trait's
  meta-learner all bases × all traits inflated Train-CV to 0.339 but dropped Dev
  to 0.276. Rejected; the honest config uses per-trait same-trait meta features
  (cross_trait=False): OOF 0.297 / Dev 0.317.
- **meta_alpha barely matters** over 0.5–8 with the small same-trait meta-feature
  set.

## Selection-criterion lesson (meta-negative-result)
- We initially let **Dev** tip a close call (core vs core+diversity): Dev favored
  the simpler **core** (0.3172 vs 0.3148) while repeated **Train-CV** favored
  **core+diversity** (0.3044 vs 0.2971). We froze **core** and it scored Test
  0.2534. Because Train-CV proved the better Test proxy here (gap 0.047 vs Dev's
  0.067), Train-CV should have been the primary tiebreak. This is the single most
  consequential methodological miss in the project.

## Not attempted (blocked / out of scope)
- **LLM-as-extractor via API** (refs [10]–[12]): an `ANTHROPIC_API_KEY` was
  present but returned HTTP 401 (invalid). A local 7–8B instruct model on the
  12 GB GPU is feasible but was deprioritized vs the proven embedding+SVR lever.
- **External corpora / augmentation / pseudo-labeling**: not needed to reach the
  current level; would be the next lever for Openness.
