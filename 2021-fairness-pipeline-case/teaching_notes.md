# Teaching Notes (Lay-reader friendly)

## The main teaching idea
Fairness can be introduced at different points in the pipeline:
- during scoring (competition-style)
- during decision policy (standards-aligned)

This repo shows why many professional guidance frameworks prefer:
- **scores that are blind to protected group membership**
- **fairness auditing and governance at the evaluation/threshold stage**

## Recommended classroom flow (60–90 minutes)
1) Read the scoring function (what the competition rewards)  
2) Inspect the competition-style baseline (how fairness enters scoring)  
3) Inspect the standards-aligned pipeline (fairness enters decision governance)  
4) Run the trade-off notebook and discuss operating points

## Discussion questions
- Why does using protected status inside scoring create defensibility challenges?
- How does the chosen cut score affect both utility and AIR?
- What additional evidence would you need before operationalizing a model (job analysis, validation)?
