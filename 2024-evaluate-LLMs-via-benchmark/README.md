# SIOP 2024 ML Competition — Four Tasks, One Harness

A post-hoc reconstruction of the SIOP 2024 ML Competition, built two years after the fact to ask one question: **can a single unified pipeline keep up with four separate hand-tuned submissions?** The 2024 winners attacked each of the four tasks (empathy, interview generation, item clarity, fairness) with different models, prompts, and stacks. This repo collapses all four into one prompt-engineering harness with task-specific adapters, runs them all against the same model, and reports the comparison.

📄 [Download the presentation deck (PDF)](https://github.com/mjmonnot/siop-ml-competitions/blob/main/2024-evaluate-LLMs-via-benchmark/docs/SIOP-2024-ML-Retrospective.pdf)

The original competition: https://github.com/izk8/2024_SIOP_Machine_Learning_Competition

## Headline results

| Task          | Metric           | Place 1 (PAID) | Place 2 (Akben) | Place 3 (Hungry Llama) | Place 4 (Wonderlic) | This repo (projected) | Measured (synthetic) |
|---------------|------------------|----------------|-----------------|------------------------|---------------------|-----------------------|----------------------|
| Empathy       | accuracy         | .580           | **.608**        | .560                   | .488                | ~.55–.60              | **1.000**            |
| Interview     | avg. cosine sim. | .440           | .496            | **.512**               | .460                | ~.46–.50              | .309                 |
| Clarity       | Pearson r        | **.816**       | .676            | .740                   | .772                | ~.65–.75              | .959                 |
| Fairness      | accuracy         | **.828**       | **.828**        | .760                   | .792                | ~.78–.85              | **1.000**            |
| **Composite** | **0.25 weighted**| **.666**       | .652            | .643                   | .630                | ~.61–.67              | **.817**             |

Task-score columns above are derived by dividing each team's reported task contribution (which is `0.25 × raw_metric`) by 0.25. The "projected" range is my expected band against the official 2024 test inputs (not measured). The "measured (synthetic)" column is from a full test-split run on `scripts/make_synthetic_data.py` inputs with `gpt-4o-2024-08-06` — **not comparable** to the winner columns; see [STATUS.md](docs/STATUS.md).

## The experiment

> This repo is a post-hoc reconstruction. The 2024 competition ended in April 2024; I am writing this in 2026. The original four winning teams had between two and seven members each and worked the problem for two months. I am writing a single unified pipeline that, by design, should be *worse* than the best of the four at each task individually — and the question is by how much, and where.
>
> The framing is pedagogical. Each task notebook shows: (1) what a naive 2024 baseline would have scored, (2) what the actual winner did and why it worked, (3) the unified-harness approach I ship here, and (4) the failure modes that matter for someone trying to do this work in their own org. The 2026 perspective matters because two things became standard practice between 2024 and 2026 that the winners didn't have: structured outputs (constrained JSON) and consistent few-shot example selection by similarity. Both are folded in.

## Architecture summary

> Every task passes through the same three-stage flow: **format** (turn the raw row into a prompt with task-appropriate examples), **call** (single OpenAI chat-completion request, structured-output constrained where applicable), **parse** (extract the typed output and validate). The only thing that varies across tasks is the adapter — a small class that knows how to build the prompt for empathy vs. fairness vs. clarity vs. interview, and how to read the response back. The harness, the retry logic, the example-selection logic, and the scoring are all shared. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design rationale and where this departs from each winner's stack.

## Repo structure

    2024-llm-task-cascade/
    ├── README.md                              ← you are here
    ├── CHANGELOG.md                           Version history (v0 → v1)
    ├── requirements.txt                       Python deps (openai, pandas, scikit-learn, sentence-transformers)
    ├── src/
    │   ├── __init__.py
    │   ├── harness.py                         Shared OpenAI client, retries, caching, structured-output helpers
    │   ├── examples.py                        Few-shot example selection (random + similarity)
    │   ├── adapters.py                        EmpathyAdapter, InterviewAdapter, ClarityAdapter, FairnessAdapter
    │   ├── scoring.py                         Per-task metrics: accuracy, cosine similarity, Pearson r
    │   └── run.py                             CLI entry point (run a task end-to-end)
    ├── scripts/
    │   └── make_synthetic_data.py             Generate plausible input CSVs when the official data isn't available
    ├── notebooks/
    │   ├── 01_empathy.ipynb                   Walkthrough + PAID Team reconstruction + unified-harness alternative
    │   ├── 02_interview.ipynb                 Walkthrough + Hungry Llama (Big-5 conditioning) reconstruction + alternative
    │   ├── 03_clarity.ipynb                   Walkthrough + PAID DeBERTa pivot story + LLM-only alternative
    │   ├── 04_fairness.ipynb                  Walkthrough + PAID Team reconstruction + unified-harness alternative
    │   └── 05_comparison.ipynb                Final scorecard: my pipeline vs. all four winners, all four tasks
    ├── data/
    │   ├── dev.csv                            Public dev-set labels (provided by competition organizers)
    │   ├── test.csv                           Public test-set labels (provided by competition organizers)
    │   └── README.md                          How to obtain the input text files (NOT in this repo)
    └── docs/
        ├── ARCHITECTURE.md                    Why one harness, when adapters need to differ
        ├── KNOWN_LANDMINES.md                 Eight traps I hit (or avoided) building this
        ├── WINNERS_SYNTHESIS.md               Side-by-side comparison of all four winning approaches
        ├── STATUS.md                          What's been run, what's pending input data
        └── ADAPTING_TO_NEW_TASKS.md           How to add a fifth I-O task to the harness

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Single task, single row (good for prompt iteration):
python -m src.run --task empathy --row-id 95 --split dev

# Full task, full split (real submission):
python -m src.run --task empathy --split test --output submissions/empathy_test.csv

# All four tasks, full submission:
for task in empathy interview clarity fairness; do
  python -m src.run --task $task --split test --output submissions/${task}_test.csv
done
python -m src.run --merge --output submissions/final.csv
```

For an end-to-end narrated walkthrough of a single task, open the corresponding notebook:

```bash
jupyter notebook notebooks/01_empathy.ipynb
```

## Input data

This repo ships only the *label* files (`dev.csv`, `test.csv`) — the actual input text (emails, interview Q&A, personality items, policy pairs) was distributed by the SIOP organizers to competition participants via the EvalAI portal and is not in this repo. See [data/README.md](data/README.md) for what filenames the harness expects and where to look for them.

If you don't have the official inputs, the repo ships a **synthetic data generator**:

```bash
python scripts/make_synthetic_data.py
```

This produces all 12 expected input files with rows that join correctly to the public label files. The pipeline runs end-to-end against them, which is sufficient to validate the engineering, demo the approach, or estimate cost. **Scores against synthetic data are not comparable to the 2024 winners' published numbers** — see [STATUS.md](docs/STATUS.md) for what synthetic-data scores tell you and don't tell you.

## Hardware

There is no hardware story here. Everything runs against the OpenAI API. A laptop with a network connection is sufficient. Wall-clock cost dominates: a full test-split run across all four tasks is roughly 270 API calls + the few-shot context (which inflates token count meaningfully on empathy and fairness, where the few-shot example set is large).

Total cost estimate against `gpt-4o-2024-08-06` at current pricing: well under \$5 for a full run, dominated by the empathy and fairness tasks where each request carries the full few-shot training set.

## Choice of model

The 2024 winners ran on `gpt-4-0125-preview` (PAID, Wonderlic), Mistral-7B / Mixtral-8x7B (Hungry Llama), and a mix of open-weight models (Akben). The harness in this repo targets `gpt-4o-2024-08-06` by default, which is the model someone would reach for in 2026 to reproduce 2024-era results most cheaply. The model name is a CLI flag — use `--model gpt-4-0125-preview` to match the winners exactly. Reasoning models (o1, o3) are deliberately not used; they're a different cost/latency regime and would change the comparison.

## Tests

```bash
python -m src.harness --selftest    # prints a row from each adapter, no API calls
python -m src.scoring --selftest    # checks all four metric implementations on canned data
```

Both should exit 0. There is no pytest harness; the goal is auditable code, not test coverage.

## License & credits

- **Pipeline:** Matthew J. Monnot, PhD
- **Competition data and benchmarks:** SIOP 2024 ML Competition organizers (Marin, Hernandez, Thompson, Yankov, Mirando), Virginia Tech, and DDI. Used here for educational reconstruction.
- **Winners whose approaches are reconstructed:** PAID Team (Jia, Son, Lee — George Mason), Akben & Aaron (Akben, Satko — Elon), Hungry Llama (Gibson, Halder, Hoffman, Johnson, Luchman, McCann, Tran — Fors Marsh), Wonderlic ML (Menchetti, Cleary, Brinza — Wonderlic). All approach descriptions are paraphrased from the public competition decks; specific prompt wordings are reproduced only as short fragments needed for the teaching notes.
- **Models:** OpenAI `gpt-4o`. Token counts and pricing accurate as of the date in CHANGELOG.

## Further reading

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — why one harness across four tasks, and where the abstraction breaks down.
- [KNOWN_LANDMINES.md](docs/KNOWN_LANDMINES.md) — eight traps that cost competition teams real score points.
- [WINNERS_SYNTHESIS.md](docs/WINNERS_SYNTHESIS.md) — side-by-side comparison of the four winning approaches, task by task.
- [STATUS.md](docs/STATUS.md) — what's actually been executed against real data and what is still pending.
- [ADAPTING_TO_NEW_TASKS.md](docs/ADAPTING_TO_NEW_TASKS.md) — how to bolt a fifth task onto the harness.
