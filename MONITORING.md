# Monitoring runbook

Labels (attended or not) arrive the day after each appointment, so inputs are monitored as well as outcomes. Alarm levels below are starting assumptions to tune after a pilot, not measured values.

| Check | Frequency | Alarm | Action | Owner |
|---|---|---|---|---|
| API errors and latency (`/health`, request logs) | Continuous | Error rate above 1% or p95 above 2 s | Restart service, check logs, roll back to the previous image if it persists | IT / platform support |
| Schema and null rates of inputs | Daily | New missing pattern, out-of-range age, new category, or bookings arriving after 17:00 in the scored list | Fix the upstream feed, hold the call list until fixed | Data steward |
| PSI on top features (`evaluate.psi_table`, training vs last 4 weeks) | Weekly | Any feature above 0.25 | Investigate source of change, review slice results, consider retraining | ML owner |
| Flag rate and mean predicted probability | Weekly | Flag rate more than 30% away from the trailing 4-week average | Check inputs first, then the threshold | ML owner |
| Realised performance (precision at flag rate, lift, no-shows reached per 100 calls) | Weekly, once labels mature | Lift below 1.25 for two weeks in a row | Recalibrate threshold or retrain | ML owner with front-desk coordinator |
| Slice performance (gender, age band, neighbourhood, lead band) | Monthly | Recall gap between groups grows beyond the gap reported at launch | Review features and threshold for the affected group | ML owner |

## When to retrain

- Lift stays below the charter target for two consecutive weeks, or any top feature stays above PSI 0.25 for two consecutive weeks.
- At least one full quarter of new labelled appointments exists. Retrain with the same `python src/train.py` and compare against the registered `final` model in MLflow before replacing it.

## When to roll back

- Lift falls below 1.10, or precision at the flag rate falls below the lead-time rule (the charter kill criteria).
- Action: stop using model scores and return to the rule (call the longest-lead bookings first) until a retrained model passes the charter criteria. Reload the previous registered version by run ID.
