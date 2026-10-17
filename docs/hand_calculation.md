# Worked hand calculation (used by tests/test_simulator.py)

Inputs: T_hot metal = 1480 C, T_water = 40 C, film h = 30 W/m2K,
skull 0.20 m (k = 30), ceramic 0.30 m (k = 5), carbon 1.50 m (k = 20), backing R = 0.02 m2K/W.

1. Resistances (m2K/W): film 1/30 = 0.03333; skull 0.20/30 = 0.00667; ceramic 0.30/5 = 0.06;
   carbon 1.50/20 = 0.075; backing 0.02. Total = 0.195.
2. Heat flux q = (1480 - 40) / 0.195 = 7384.6 W/m2.
3. Temperature drop in each layer = q x R: film 246.2, skull 49.2, ceramic 443.1, carbon 553.8,
   backing 147.7. Sum = 1440.0 C (= 1480 - 40), which proves the same flux crosses every layer.
4. Thermocouple at distance x from the cold face of the carbon: T = 40 + q x (0.02 + x/20)
   x = 0.2 m -> 261.5 C;  x = 0.4 m -> 335.4 C;  x = 0.6 m -> 409.2 C.
