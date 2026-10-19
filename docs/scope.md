# Project scope (Week 1)

**Working title (to confirm):** Sensor-Robust Physics-Informed Estimation of Blast Furnace Hearth Refractory Wear: A Simulation-Based Study

## Question
Can a neural network that is trained with heat-transfer physics estimate the remaining hearth wall thickness and protective skull thickness from a few thermocouple readings, and does it cope better than a data-only network when thermocouples fail?

## Approach
1. Build a 1D heat-transfer simulator of a hearth sidewall (skull, ceramic wall, carbon blocks, backing, cooling) that produces thermocouple readings together with the true layer thicknesses.
2. Generate 200 furnaces of 730 days each, with noise and sensor dropouts.
3. Compare three estimators on furnaces never seen in training: a classical 1D inversion (industry method), a data-only LSTM or GRU, and a physics-informed version with a forward-consistency loss and a no-regrowth penalty.
4. Main experiment: switch off 1 or 2 of the 3 thermocouples and compare how much each estimator degrades.
5. Extra check: test the trained models on data from a second simulator variant (different conductivity behaviour) to show they do not just memorise one formula.

## What is real and what is simulated
- **Simulated:** all thermocouple readings and all layer thicknesses. No real hearth thermocouple data was available.
- **From published sources:** initial wall dimensions, normal wear rates (16 to 38 mm/year, derived from field tables), carbon block conductivity range, hot metal temperature range, the 1150 C wear-surface isotherm. See `parameters.csv` for the source of each value.
- **Assumed or tuned:** ceramic conductivity, backing resistance, water temperature, the effective hot-side heat transfer coefficient, skull growth and melt rates, thermocouple depths. Each is marked in `parameters.csv`.
- **Real plant data (Mendeley):** to be used only for realistic operating variation and sensor-fault patterns, not as lining data.

## Success criteria
- The physics-informed model has lower error than the classical inversion and the data-only network on unseen furnaces.
- With 1 or 2 thermocouples masked, the physics-informed model loses less accuracy than the data-only model. Results are reported over several random seeds.
- Everything is reproducible with one command from the GitHub repository.

## What this project will not claim
- It has not been validated on a real furnace, and it has not been reviewed by a domain expert.
- It is a 1D slice, not a full 2D or 3D wear map.
- It is not real-time monitoring: it replays simulated histories.

## Known limitations so far
- The skull rule relies on a tuned hot-side heat transfer coefficient (20 to 40 W/m2.K) for which no published value was found, so the skull mechanism is the least well supported part of the model.
- Normal wear moves the sensors only by tens of degrees over two years, which is comparable to operating variation.
- The skull-growth rates, the protective effect of a skull on wear and the effective heat transfer coefficient are assumptions. Skull conductivity (2 W/m.K) comes from one published study.
