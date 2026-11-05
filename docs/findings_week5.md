# Week 5 findings: the data-only GRU and the choice of its settings

Code: `simulator/train_gru.py`, `simulator/sweep_gru.py`. Tests: `tests/test_gru.py`, `tests/test_sweep.py`.
All scores are on the validation split (30 unseen furnaces). The test sets have not been used. All data is simulated.

## First three-seed result (window 60 days, hidden size 32)
| Model | Skull change | 1150 C distance change | Wall change |
|---|---|---|---|
| GRU, data only (R2, mean of 3 seeds) | 0.80 | 0.47 | 0.42 |
| Boosted trees on a window summary | 0.78 | 0.18 | 0.20 |
| Ridge on a window summary | 0.61 | 0.28 | 0.26 |

## Choosing the settings (decided before the physics-informed network exists)
Rule, fixed in advance: best mean validation R2 over the three targets, averaged over 2 seeds; within 0.01 of the best, the simplest
(smallest window length x hidden size) wins. All settings are scored on the same validation days (those a 120-day window can reach).

| Window | Hidden | Skull | 1150 C | Wall | Mean R2 |
|---|---|---|---|---|---|
| 30 | 16 | 0.803 | 0.432 | 0.401 | 0.545 |
| 30 | 32 | 0.806 | 0.448 | 0.417 | 0.557 |
| 30 | 64 | 0.802 | 0.447 | 0.411 | 0.553 |
| 60 | 16 | 0.811 | 0.454 | 0.423 | 0.562 |
| 60 | 32 | 0.803 | 0.483 | 0.452 | 0.579 |
| **60** | **64** | 0.811 | 0.499 | 0.465 | **0.592** |
| 120 | 16 | 0.824 | 0.386 | 0.438 | 0.549 |
| 120 | 32 | 0.820 | 0.442 | 0.457 | 0.573 |
| 120 | 64 | 0.831 | 0.456 | 0.485 | 0.590 |

**Chosen: window 60 days, hidden size 64.** The 120 / 64 setting is within the tolerance (0.590), so the rule prefers the shorter window.

## How much weight this choice can carry
- Differences between neighbouring settings are small. Two seeds of the same setting differ by up to 0.06 in the 1150 C score
  (for example 0.425 and 0.487 at window 120, hidden 64), which is as large as most gaps in the table.
- What the table does support: a very short window (30 days) or a very small network (16) is somewhat worse than 60 to 120 days with 32 to 64 units.
- The choice is a reasonable fixed setting, not a proven optimum. It was made before any physics-informed result exists, which is the point.

## Rules for Week 6
Both networks (data-only and physics-informed) use window 60 and hidden size 64, the same optimiser, epochs, early-stopping rule and the same
random sensor-masking during training. Then any difference between them comes from the physics, not from settings or from the masking.
