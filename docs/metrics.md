# Competition metric: mean Pearson r

For each trait:

1) Compute Pearson correlation **r** between the true scores and predicted scores.
2) Average r across traits.

## Why Pearson r is a little “weird” (and important)
- Pearson r evaluates **rank-order agreement**, not absolute calibration.
- A model can have decent RMSE but low r if it predicts everyone near the mean (variance shrinkage).
- Conversely, a model can have a higher r even if its predictions are not perfectly calibrated.

## Practical implication for modeling
If the scoring metric is r:
- You often want models that preserve *relative differences* between people.
- Over-regularized models can underperform if they collapse variance too much.

