# Data folder (not tracked)

Place competition CSVs here (gitignored):

| File | Role |
|------|------|
| `train.csv` | Labeled training data (**required** for `python -m src.run_compare`) |
| `participant_dev.csv` | Unlabeled public dev features (optional) |
| `participant_test.csv` | Unlabeled public test features (optional; for submission-style runs) |

## How to obtain

1. Upstream winners / data release:
   https://github.com/izk8/2021_SIOP_Machine_Learning_Winners/tree/main/data
2. Or copy from a local archive folder named `2021 Winners and Data/`
   (kept next to this case for authoring; also gitignored).

```powershell
copy "2021 Winners and Data\train.csv" "00_data\train.csv"
copy "2021 Winners and Data\participant_dev.csv" "00_data\participant_dev.csv"
copy "2021 Winners and Data\participant_test.csv" "00_data\participant_test.csv"
```

## Label note

Only ~18% of `train.csv` rows have `High_Performer` / `Overall_Rating`.
The freeze comparison keeps rows with non-null
`High_Performer`, `Retained`, and `Protected_Group` (n≈7,890).
Dev/test participant files have **no** outcome labels — private-test
scores cannot be recomputed.
