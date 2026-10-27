# Sensor fault study (Week 3)

Code: `simulator/sensor_fault_study.py`. Data: Mendeley "Process data collected by blast-furnace sensors"
(29,602 hourly rows, 1 Jan 2013 to 18 May 2016, 1,233 days, no timestamp gaps).

**This is a proxy.** The 15 temperature columns (Tt1 to Tt4, Tp1 to Tp10, Tc) are upper-furnace gas and air
temperatures, not thermocouples buried in the hearth lining. Hearth thermocouples may fail differently.

## What the real sensors do
- No cell is blank. Faults appear as impossible numbers: the most common is 3311 (also 3310, 3314, 3309),
  and some readings are negative.
- 56 fault events in 15 sensors over 1,233 days: about 0.30 events per sensor per 100 days.
- Mean event length about 25 hours (1.0 day). One sensor (Tp4) was out for 506 hours (about 21 days) in 2016.
- In 787 hours (2.7%) at least one sensor was faulty. In 678 of those hours only one sensor was; in 22 hours
  all 15 failed at once, which points to a logging fault rather than a sensor fault.
- Noise ceiling: the smoothest sensor (Tc) changes by a robust 0.78 C from one hour to the next, and that
  includes real process change, so true sensor noise is at most about that. The gas temperature sensors change by
  10 to 24 C an hour, which is real process variation, so they say nothing about noise.

## What the simulator assumed before, and now
| Setting | Before | Now | Basis |
|---|---|---|---|
| Outage starts (per sensor per day) | 0.010 | 0.003 | 0.30 events per 100 days |
| Typical outage length | 1 to 5 days (mean 3) | median 1 day (lognormal), 90th percentile 2 days | mean 1.0 day |
| Long outages | none | 2% of outages last 8 to 21 days | one 21-day outage in 56 events |
| Outages hitting all three sensors together | none | 0.001 per day, lasting about a day | 22 hours with all 15 down |
| Noise (standard deviation) | 1.5 C | 1.5 C (kept) | ceiling of 0.78 C on one smooth sensor; 1.5 C is deliberately conservative |

Readings during an outage are stored as blanks (NaN), as they would be after cleaning, instead of as 3311.

## Limits
The fault shapes come from sensors of a different kind in a different part of the furnace. The experiment that
switches off 1 or 2 thermocouples on purpose is separate from this and does not depend on these rates.
