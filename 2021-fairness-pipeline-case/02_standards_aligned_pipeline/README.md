# Standards-aligned pipeline (fully annotated)

This folder teaches a pipeline pattern that is more defensible for applied hiring:

1) **Prediction step** (model scoring):
   - predict a business-relevant outcome using job-related predictors
   - do NOT use protected-group membership in scoring

2) **Decision step** (policy governance):
   - choose a threshold using documented trade-offs
   - compute AIR and other fairness metrics using protected-group membership **only for evaluation**

Files:
- `scoring_function.py` — implements AIR, unfairness, and the competition score
- `job_success_model.py` — predicts a Job Success Index (aligned to business priorities)
- `cutoff_optimization.py` — selects a cut score via trade-off search and outputs Hire/No-hire
