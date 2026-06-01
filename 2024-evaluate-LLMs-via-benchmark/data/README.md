# Data Directory

This directory contains the **label files** for the 2024 SIOP ML Competition:

- `dev.csv` — 314 rows, ground-truth labels for the public dev set across all four tasks
- `test.csv` — 273 rows, ground-truth labels for the public test set across all four tasks

Both files have the same schema: `benchmark,_id,output`, where `benchmark` is one of `empathy`, `interview`, `clarity`, `fairness`, `_id` is the row identifier (numeric for empathy/clarity/fairness, Qualtrics-style like `R_xxx` for interview), and `output` is the ground-truth label/value as a string.

## What's NOT here

The **input text files** for each task. These were distributed to competition participants via the [EvalAI portal](https://eval.ai/web/challenges/challenge-page/2207/overview). They are not in this repo because (a) my post-hoc reconstruction doesn't have them either and (b) you should get them from the source rather than a downstream copy.

The expected filenames are:

| Task      | Train                  | Dev (validation)             | Test                        | Required columns |
|-----------|------------------------|------------------------------|-----------------------------|------------------|
| empathy   | `empathy_train.csv`    | `empathy_val_public.csv`     | `empathy_test_public.csv`   | `_id, text, empathy` (train); `_id, text` (dev/test) |
| interview | `interview_train.csv`  | `interview_val_public.csv`   | `interview_test_public.csv` | `_id, Q1..Q5, R1..R5` (train); `_id, Q1..Q5, R1..R4` (dev/test) |
| clarity   | `clarity_train.csv`    | `clarity_val_public.csv`     | `clarity_test_public.csv`   | `_id, item, mean_clarity` (train); `_id, item` (dev/test) |
| fairness  | `fairness_train.csv`   | `fairness_val_public.csv`    | `fairness_test_public.csv`  | `_id, first_option, second_option, majority_vote` (train); `_id, first_option, second_option` (dev/test) |

The `_id` column in each input file should match the `_id` column in the corresponding label file (`dev.csv` or `test.csv`).

## File naming

The CLI in `src/run.py` accepts either of two naming conventions:

- The competition convention: `{task}_train.csv`, `{task}_val_public.csv`, `{task}_test_public.csv`
- A simpler convention: `{task}_train.csv`, `{task}_dev_inputs.csv`, `{task}_test_inputs.csv`

Whichever exists is picked up automatically.

## How to obtain

Three options:

1. **From the SIOP organizers.** The data is publicly released; the competition's GitHub repo (https://github.com/izk8/2024_SIOP_Machine_Learning_Competition) points to the data files in its `00 - data release/` directory.

2. **From the EvalAI portal.** https://eval.ai/web/challenges/challenge-page/2207/overview — register for the (now-closed) competition and download the data files.

3. **From the winners' published reproduction notebooks.** PAID Team's Colab notebooks (in the upstream competition repo) reference Google Drive paths to the input files; their published notebook would either include the files or point at them.

## Self-test

Once you have the input files in place, sanity-check the joins:

```python
import pandas as pd
test_labels = pd.read_csv("data/test.csv", encoding="utf-8-sig")
empathy_inputs = pd.read_csv("data/empathy_test_public.csv", encoding="utf-8-sig")

# Every empathy test label should have a matching input row:
empathy_test_ids = set(test_labels.query("benchmark == 'empathy'")._id.astype(str))
empathy_input_ids = set(empathy_inputs._id.astype(str))
assert empathy_test_ids <= empathy_input_ids, "missing inputs for some test rows"
```

If that passes, the harness will be able to find every test row's input.

## Submissions

This directory is also where merged submission files go if you run the pipeline end-to-end. The expected output filename is whatever you pass to `--output` (e.g., `submissions/final.csv`). The merged format matches the EvalAI portal's submission template: `benchmark,_id,output` with one row per (task, item) pair.
