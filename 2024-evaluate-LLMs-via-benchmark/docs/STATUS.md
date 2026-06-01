# Status

What's actually executable in this repo, and what's not.

## TL;DR

The pipeline is **end-to-end runnable** against synthetic inputs that the repo can generate on demand. It is **not** end-to-end runnable against the official 2024 SIOP competition inputs, because those files were distributed via the EvalAI portal and are not publicly archived. The label files (`dev.csv`, `test.csv`) — which contain the ground-truth answers — are public and shipped with this repo, but they're useless without the corresponding input files (the actual emails, interview Q/R pairs, personality items, and policy descriptions).

This means **scores from this repo are not directly comparable to the 2024 winners' published numbers.** They tell you whether the *pipeline* works — prompt construction, API dispatch, parsing, scoring, submission file writing. They don't tell you how this approach would have placed in the actual competition.

## What runs

| Component                          | Status                  | Notes                                                            |
|------------------------------------|-------------------------|------------------------------------------------------------------|
| `src/harness.py`                   | ✅ Self-test passes     | `python -m src.harness --selftest`                               |
| `src/examples.py`                  | ✅ Self-test passes     | `python -m src.examples --selftest`                              |
| `src/scoring.py`                   | ✅ Self-test passes     | Includes a check that the composite formula reproduces PAID's .666 |
| `src/adapters.py`                  | ✅ Self-test passes     | All four adapters round-trip canned inputs                       |
| `src/run.py`                       | ✅ Measured on synthetic | Full dev + test sweeps executed 2026-05-31; see below            |
| `scripts/make_synthetic_data.py`   | ✅ Generates all 12 input CSVs | Joins cleanly to `dev.csv` / `test.csv` ground-truth labels |
| Label files (official)             | ✅ Present              | `data/dev.csv` (314 rows), `data/test.csv` (273 rows)            |
| Input files (official)             | ❌ Not in this repo     | Were distributed via the EvalAI portal; not publicly archived    |
| Input files (synthetic)            | ✅ Generated on demand  | Run `python scripts/make_synthetic_data.py`                      |
| `notebooks/01_empathy.ipynb`       | ⚠️ Pedagogical          | Code runs against the harness; scores are synthetic-data scores  |
| `notebooks/02_interview.ipynb`     | ⚠️ Pedagogical          | Same                                                             |
| `notebooks/03_clarity.ipynb`       | ⚠️ Pedagogical          | Same                                                             |
| `notebooks/04_fairness.ipynb`      | ⚠️ Pedagogical          | Same                                                             |
| `notebooks/05_comparison.ipynb`    | ✅ Updated              | Headline table includes measured synthetic-data scores           |
| `submissions/`                     | ✅ Present              | Dev + test CSVs (+ SC5 variants for empathy/fairness)            |

## Measured synthetic-data scores

Executed 2026-05-31 against synthetic inputs with `gpt-4o-2024-08-06`, default harness settings, `.harness_cache/` enabled. **These numbers are not comparable to the 2024 winners' published scores** — the synthetic inputs are label-correlated templates, not the official EvalAI text.

### Test split (primary)

| Task     | Metric              | Score  |
|----------|-----------------------|--------|
| Empathy  | accuracy              | 1.000  |
| Interview| avg cosine similarity | 0.309  |
| Clarity  | Pearson r             | 0.959  |
| Fairness | accuracy              | 1.000  |
| **Composite** | 0.25-weighted    | **0.817** |

### Dev split

| Task     | Metric              | Score  |
|----------|-----------------------|--------|
| Empathy  | accuracy              | 1.000  |
| Interview| avg cosine similarity | 0.292  |
| Clarity  | Pearson r             | 0.962  |
| Fairness | accuracy              | 1.000  |
| **Composite** | 0.25-weighted    | **0.814** |

### Self-consistency (N=5, test split)

Empathy and fairness with `--self-consistency 5`: both unchanged at 1.000 (synthetic data is already at ceiling; no marginal gain from majority vote).

### Landmines observed

- **Empathy template clarity** — accuracy 1.000 on both splits (expected inflation).
- **Fairness orientation** — accuracy 1.000 on both splits (expected inflation).
- **Clarity LLM ceiling** — r=.959/.962, well above the ~.70 real-data ceiling (synthetic inflation, not a pipeline bug).
- **Interview cosine** — cos=.292/.309, below winner range (synthetic R1–R4 uncorrelated with ground-truth R5; data limitation).

## Synthetic data: what it is and isn't

The generator in `scripts/make_synthetic_data.py` produces 12 CSV files matching the expected schema:

- 4 training files (one per task)
- 4 dev input files (`{task}_dev_inputs.csv`)
- 4 test input files (`{task}_test_inputs.csv`)

Each row in the dev/test input files uses the **exact `_id` from the official `dev.csv` / `test.csv`**, so the join keys work and the scoring code runs unchanged. The synthetic *content* is generated to correlate with the ground-truth label: an empathy email labeled 1 is built from "empathetic" template fragments, one labeled 0 from "cold" fragments; a fairness pair labeled "first" has the more supportive option in position 1; etc.

**Good for:**

- Confirming the pipeline runs end-to-end (prompt construction, API calls, parsing, scoring, CSV writing)
- Getting a cost estimate before paying for the real run
- Sanity-checking that the adapters return sensibly-typed outputs
- Demonstrating the approach in a presentation when the official data isn't available

**Not good for:**

- Apples-to-apples comparison against the 2024 winners. The winners ran their pipelines against specific inputs; we don't have those inputs. Synthetic-data scores measure the pipeline's behavior on *our* inputs, which are easier than the official ones (the synthetic empathy/fairness data has no label noise; the official benchmarks had substantial human-rater disagreement).
- Drawing any conclusion about the difficulty of the 2024 benchmark. A pipeline that scores .95 on synthetic empathy might score .55 on the real benchmark.

Expected synthetic-data scores: empathy and fairness will probably land near 1.0 (the templates are too easy); clarity will be closer to real performance (the items are real-looking statements drawn from public personality inventories); interview cosine will be low (R1-R4 are synthesized independently of the ground-truth R5, so the model can't generate a stylistically-matched response).

## What's missing for a real comparison

The official input text files. Expected filenames (from the PAID Team notebooks and Hungry Llama documentation):

- `empathy_train.csv`, `empathy_val_public.csv`, `empathy_test_public.csv`
- `interview_train.csv`, `interview_val_public.csv`, `interview_test_public.csv`
- `clarity_train.csv`, `clarity_val_public.csv`, `clarity_test_public.csv`
- `fairness_train.csv`, `fairness_val_public.csv`, `fairness_test_public.csv`

See `data/README.md` for the schema each adapter expects. If you find these files (your own archives from 2024, the upstream competition repo, or by reaching out to the organizers), drop them in `data/` and re-run — they take precedence over the synthetic files.

## What the headline-results "projected" column means

The numbers in the README's headline table marked as "this repo (projected)" are NOT measured scores. They are an honest prior estimate based on the four winners' published per-task scores plus the architectural choices in this repo (LLM-only on clarity, full few-shot on fairness, similarity-selected few-shot on empathy, no-external-examples on interview, structured outputs throughout, self-consistency optional).

The midpoint of each range is roughly where I'd expect this pipeline to land on the actual 2024 test set if the inputs were available. The actual scores could land outside the bands; that's what running real numbers would tell you.

## How to get real numbers

1. Obtain the input text files (see above).
2. Drop them in `data/`. The runner picks them up automatically — both naming conventions (`{task}_test_public.csv` and `{task}_test_inputs.csv`) are accepted.
3. Run all four tasks against the test split:

   ```bash
   for task in empathy interview clarity fairness; do
     python -m src.run --task $task --split test \
       --output submissions/${task}_test.csv
   done
   python -m src.run --merge submissions/{empathy,interview,clarity,fairness}_test.csv \
     --output submissions/final.csv
   ```

4. Update the headline table in README.md with the measured numbers.
