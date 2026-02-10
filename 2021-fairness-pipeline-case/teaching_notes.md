# Teaching Notes: Fairness-in-the-Pipeline (SIOP 2021 Case)

## Recommended teaching flow (60–90 minutes)
1. **Read the competition metric** (5–10 min): emphasize subtractive unfairness penalty and heavy weight on retained top performers.
2. **Walk through the reference solution** (15–20 min): identify fairness insertion point (training/scoring) and decision rule (median cut).
3. **Introduce professional framing** (10–15 min): “protected class variables used for evaluation, not scoring.”
4. **Run Standards-aligned approach** (15–25 min): show AIR/accuracy trade-off curves and threshold choice.
5. **Discussion** (10–20 min): transparency, defensibility, documentation, stakeholder communication.

## Key discussion questions
- Why does moving fairness from *scoring* to *thresholding* change defensibility?
- What are the benefits and risks of fixing a selection rate (e.g., 50%)?
- What additional evidence would you need for job relatedness (job analysis, construct mapping)?
- How would you communicate trade-offs to HR/legal leaders?

## Framing language you can reuse
- “Protected-group membership is used to **evaluate outcomes**, not to **generate scores**.”
- “We treat the cut score as a governance decision: it should be chosen transparently and documented.”
- “We separate prediction (validity) from decision policy (fairness and risk tolerance).”

## Instructor tips
- Keep the competition baseline as a *contrast case* (not a recommendation).
- Use the trade-off curve figure to show that “fairness” is not a single number; it depends on operating point.
- Encourage students to propose alternatives (reweighting, constraint-based optimization, separate decision policies).
