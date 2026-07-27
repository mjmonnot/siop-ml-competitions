# Negative results — what did not work (or should not be claimed)

## Unconstrained threshold “wins” are invalid

Allowing the standards-aligned cut search to pick any selection rate yields
~99 final score by hiring ~96% of the holdout. That is metric gaming, not a
deployable policy. Primary results constrain selection rate to ~50%.

## Standards-aligned does not beat Place 1 on the contest formula

Under a fair 50% hiring volume, the standards-aligned path ties the
no-proxy competition-style ablation and **loses** to the protected-proxy
ensemble on holdout final score. Claiming a leaderboard win for Path B would
be false. The case’s value is governance contrast, not podium conquest.

## Private-test reproduction is impossible

No amount of hyperparameter search recovers a verifiable private-test final
without labels. Projected or “estimated” private-test scores are not reported
as measured.

## AIR≥0.80 did not change the operating point

On this freeze split, the metric-optimal ~50% threshold already had AIR≈0.84,
so the AIR-floor variant selected the same cut. That is a property of *this*
split/model, not a general guarantee that 4/5ths constraints are free.

## Full winner re-implementation was out of scope

Axiom (tidymodels racing) and RHDS (R) are synthesized from public materials
and decks, not byte-for-byte re-run under our freeze protocol. Only the
Procrustination-style Python recipe and the standards-aligned alternative are
measured end-to-end here.
