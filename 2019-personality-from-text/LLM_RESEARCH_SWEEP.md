# LLM Research Sweep

Ran stages 1-5 on Train-CV + Dev only. No Test evaluation was run in this sweep.

## Results

| Stage | Description | OOF mean r | Dev mean r | Notes |
|---|---:|---:|---:|---|
| Current | Haiku general + e5 SVR + TF-IDF + engineered | 0.3355 | 0.3474 | Existing winning config |
| 1 | Add Haiku evidence + ranked prompt variants | 0.3421 | 0.3414 | OOF up, Dev down |
| 2 | Add Haiku trait_focus prompt | 0.3431 | 0.3407 | OOF up, Dev down |
| 3 | Add Sonnet 4.6 judge | 0.3465 | 0.3465 | Neutral/slightly below current |
| 4 | Add Haiku behavioral subfeatures | **0.3627** | **0.3548** | Best Train-CV/Dev candidate |
| 5 | Rank-oriented meta layer over stage 4 | 0.3525 | 0.3381 | Worse than Ridge meta |

## Interpretation

The only clear improvement over the existing frozen stack is stage 4: Haiku behavioral subfeatures. The feature base alone is strong (OOF 0.3131, Dev 0.3081), and the full stack moves Dev from 0.3474 to 0.3548.

Prompt self-consistency, trait-focus prompting, and Sonnet add diversity but do not improve Dev on their own. Rank-meta is worse than the standard Ridge meta layer.

## Recommended candidate

Use stage 4 for the next frozen Test candidate if we decide to spend another Test evaluation:

- Haiku general scores
- Haiku evidence/ranked/trait_focus scores
- Sonnet general scores
- Haiku behavioral subfeatures
- e5-large per-prompt SVR
- e5-large all-text SVR
- TF-IDF char + word-unigram
- engineered + engineered_gbm

Log: `results/llm_research_sweep.log`
