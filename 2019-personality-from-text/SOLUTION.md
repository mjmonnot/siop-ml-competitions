# Solution write-up - 2019 SIOP ML (personality from open-ended text)

## Headline result (Stage 5 frozen Test evaluation)

| Split | mean r | A | C | E | N | O |
|---|---|---|---|---|---|---|
| Train-CV (OOF) | **0.3635** | 0.434 | 0.323 | 0.445 | 0.319 | 0.296 |
| Dev (public) | **0.3600** | 0.493 | 0.226 | 0.462 | 0.250 | 0.369 |
| **Test (private)** | **0.3215** | 0.387 | 0.283 | 0.410 | 0.254 | 0.274 |

**2019 first place (Test): 0.26021 -- BEATEN by +0.061.**

Stage 5 improves on Stage 4 (Test 0.3175 -> **0.3215**, +0.004) by adding a role-play
questionnaire base (`llmq:`): Haiku answers a 30-item BFI-2-style battery in the
respondent's persona, reverse-scored and aggregated to trait scores.

## What drives the gain

The winning stack combines:

- Haiku 4.5 Big Five score extractor (4 prompt variants: general, evidence, ranked, trait-focus)
- Sonnet 4.6 Big Five score extractor (second judge)
- Haiku behavioral subfeature extractor (16 dimensions -> Ridge head)
- **Haiku role-play questionnaire aggregate (`llmq:`)** -- new in Stage 5
- e5-large-v2 per-prompt SVR and all-text SVR
- TF-IDF char + word-unigram Ridge
- Engineered psycholinguistic features (Ridge + HistGBM)

Per-trait Ridge meta-learner (alpha=4, own-trait columns only).

The questionnaire lever validated on Dev (+0.005 mean r, 0.3548 -> 0.3600) and generalized
on Test (+0.004). A two-stage persona-summary variant (Liu et al. 2025) was piloted but
did not beat the plain questionnaire aggregate on mean r; not included in the frozen model.

## Reproduce

```powershell
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="your-key"
python -m src.freeze_and_test          # full Test evaluation (single touch)
python -m src.freeze_and_test --dev-only   # Train-CV + Dev only
```

Outputs:
- `results/cv/frozen_summary.csv`
- `results/submissions/submission_test_frozen.csv`
- `results/submissions/submission_dev_frozen.csv`

Note: one Sonnet Test-row call in Stage 4 returned incomplete JSON and was mean-imputed;
Stage 5 re-ran with full questionnaire cache on all splits.

## Prior milestones

| Stage | Test mean r | Key addition |
|---|---|---|
| Core (no LLM) | ~0.26 | e5-large SVR + TF-IDF + engineered |
| Stage 4 | **0.3175** | LLM direct scores + behavioral subfeatures |
| **Stage 5** | **0.3215** | + role-play questionnaire (`llmq:`) |
