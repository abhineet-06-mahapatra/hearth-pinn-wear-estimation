# Week 3 findings: the time-dependent heat model

Code: `simulator/hearth_transient.py`. Tests: `tests/test_transient.py` (11 tests; 38 in total with the earlier ones).
All numbers describe the simulator, not a real furnace.

## What it is
The wall is split into cells (15 carbon, 6 ceramic, 3 skull). Each day the heat equation is advanced in four
implicit steps, so the thermocouples respond with a lag instead of jumping to a new steady value.
It uses the same operating conditions, skull rule and wear rules as `hearth_simulator_v2.py`, and follows
identical furnace histories (checked by a test).

## Checks that passed
- With constant conditions it reproduces the series-resistance result of the steady model to 1e-6 relative error.
- The same heat flux crosses the film and the backing at steady state.
- Heat is conserved exactly in each implicit step (stored heat equals heat in minus heat out).
- After a step change the readings rise monotonically and settle to the new steady value.
- The 1150 C isotherm from the cell profile matches the analytic value (1.7766 m for the worked example).

## Results
Time for TC3 to cover 63% of a +100 C step in hot metal temperature (ceramic 0.30 m, skull 0.10 m):

| Carbon thickness | Response time |
|---|---|
| 0.7 m | 1.25 days |
| 1.0 m | 1.75 days |
| 1.5 m | 2.50 days |
| 1.9 m | 3.25 days |

So the lag does carry information about wall thickness: thicker walls respond more slowly.
Over 20 furnaces the transient TC3 reading differs from the steady-state reading by a median of 0.2 C
(5th to 95th percentile -6.7 to +4.3 C), and by up to 77 C during upsets.

## What this means
1. The lag is on a scale of 1 to 3 days, so daily readings resolve it only coarsely. A model can use it
   only if it sees how readings change after an event, and operating events are not part of its inputs.
2. For slow wear the transient and steady models agree, so the lag matters mainly for upsets and regime changes.
3. The dataset can now be generated with the transient model (about 30 s for 200 furnaces on this machine's test run).

## Assumptions to replace
- Heat capacities of carbon (2.0e6), ceramic (3.0e6) and skull (3.5e6) J/m3.K are assumptions with no source yet.
- When a layer changes thickness its cells stretch and keep their temperatures, which does not conserve heat
  exactly across that change (small for slow wear).
