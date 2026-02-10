# Standards-aligned pipeline (recommended pattern)

This pipeline aims to be more consistent with common professional expectations (e.g., transparency, governance, documentation)
by applying the following principle:

> **Protected-group membership is used to evaluate outcomes and guide decision policy,
> not to generate individual prediction scores.**

Components:
- `scoring_function.py`: Implements the competition scoring function + AIR utilities.
- `job_success_model.py`: Trains a model to predict a *Job Success Index* aligned to the business weights.
- `cutoff_optimization.py`: Chooses a cut score via a documented trade-off search (utility vs AIR),
  using protected-group membership only for evaluation.

The goal is to teach a **pipeline pattern**, not to assert that any single fairness metric is “the one true metric.”
