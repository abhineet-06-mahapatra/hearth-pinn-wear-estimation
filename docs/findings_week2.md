# Week 2 findings (sensitivity and identifiability)

All numbers come from `simulator/sensitivity_plots.py` and `simulator/extrapolation_check.py`.
They describe the simulator, not a real furnace.

## Sensitivity of the TC3 reading (reference state: ceramic 0.30 m, carbon 1.50 m, skull 0.10 m)
| Change | Effect on TC3 |
|---|---|
| 100 mm of carbon worn away | +6 C |
| 100 mm of ceramic worn away | +26 C |
| Carbon conductivity 13 to 35 W/m.K | -82 C (equal to about 1.3 m of carbon wear) |
| A 0.20 m skull melts away | +125 C |
| Backing resistance 0.02 to 0.0083 | -58 C (shifts all readings equally) |
| Skull melts as flow changes, thick wall | +175 C |
| Skull melts as flow changes, thin wall (carbon 0.80 m, no ceramic) | +579 C (looks too large, to be checked) |

## What this means
1. Carbon wear is a faint signal compared with sensor noise (1.5 C) plus the unknown conductivity.
   The absolute carbon thickness cannot be read across furnaces when conductivities differ.
2. A straight-line extrapolation from two sensors does NOT recover the 1150 C isotherm: it overshoots by
   about 1 m (median absolute error 1.06 m), because that isotherm usually lies in the ceramic or skull.
3. Three temperature readings cannot separate three layer thicknesses when conductivities also vary.
   The project therefore estimates the total remaining wall and the distance to the 1150 C isotherm
   (column `iso1150_m`), not each layer separately. The true isotherm distance tracks the remaining wall
   closely (correlation about 0.9).

## Decisions taken
- Added `iso1150_m` (distance from the carbon cold face to the 1150 C isotherm) as a label.
- Time-dependent heat model moved from optional to planned for Week 3: the response over time carries
  information about wall thickness that a steady-state reading does not.
- The classical baseline in Week 4 must use nominal layer properties, not the naive straight line.

## Open
- Thermocouple temperatures have no verified published values.
- Backing resistance (0.02 against 0.0083) is undecided.
- The thin-wall skull-melt swing needs checking against published behaviour.
