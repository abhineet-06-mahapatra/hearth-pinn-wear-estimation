# Week 4 findings: baselines and what limits them

Code: `simulator/baselines.py` (scores), `simulator/diagnostics_week4.py` (the experiments below), tests in `tests/test_baselines.py`.
All scores are on the validation split (30 unseen furnaces). The test sets have not been used. All data is simulated.

## Baseline scores, single day, absolute values (validation R2, higher is better)
| Model | Remaining wall | 1150 C distance | Skull |
|---|---|---|---|
| Mean of training data | -0.00 | -0.00 | -0.00 |
| Classical: straight-line extrapolation plus 4 fitted numbers | 0.16 | 0.30 | -0.00 |
| Ridge regression on readings and differences | 0.15 | 0.32 | 0.01 |
| Gradient-boosted trees | 0.13 | 0.27 | 0.24 |

RMSE of the best model: wall 0.40 m (the mean gives 0.43), 1150 C distance 0.33 m (0.41), skull 0.054 m (0.062).
Boosted trees fit the training furnaces much better (R2 0.44, 0.35, 0.45) than the validation furnaces, so they overfit.

## Why the scores are low
1. Knowing the carbon and ceramic conductivities and the backing resistance barely helps (wall R2 0.15 to 0.19 with a ridge model).
2. Knowing the hot-side film resistance helps more (wall 0.28, 1150 C distance 0.39) but not enough; with everything known a ridge model reaches only 0.35 to 0.43.
3. Resistance is dominated by the ceramic: 0.4 m of ceramic resists as much heat as about 1.9 m of carbon. Three sensors measure the total resistance, so
   a thick carbon wall without ceramic and a thinner one with ceramic can read the same. The absolute remaining wall is a poorly posed target.

## What works better: the change since the furnace's own start
Using each sensor's change from the median of the furnace's first 10 days as input, and the change in the label as the target:

| Inputs | Wall change | 1150 C change | Skull change |
|---|---|---|---|
| Today's readings (absolute) | 0.01 | 0.31 | 0.02 |
| Today's change since start (ridge / boosted trees) | 0.31 / 0.04 | 0.26 / 0.06 | 0.63 / 0.76 |
| Change plus 30 and 90 day history (best of two models) | 0.16 | 0.14 | 0.83 |

- The skull change is estimable: R2 0.63 to 0.83.
- The wall change stays hard (best 0.16 with history, 0.31 for one simple single-day model). Wear moves a reading by only 6 to 26 C per 100 mm,
  against 1.5 C of noise and tens to hundreds of degrees from skull and flow changes. Wear is slow and monotonic, the confounders are fast,
  so a model with a long memory and a no-regrowth rule is the natural way to improve it. This is the hypothesis the neural networks will test.

## Consequences for the project
- Headline target: skull thickness change. Secondary: 1150 C isotherm distance. Wall change is reported honestly as the hard case.
- A comparison of networks under sensor failure is informative only where the all-sensors score is decent, so the skull is the main case.
- The dataset does not need to change. Labels expressed as change since day 0 and per-furnace reference readings (first 10 days) are derived from
  existing columns. This assumes the starting state is known, as it would be after commissioning or an inspection.

## Limits
Simulated data; the baselines see single days or simple rolling summaries, not a sequence model; the model structure of the simulator
(effective hot-side film, assumed layer properties) limits how much wear information the sensors can carry.
