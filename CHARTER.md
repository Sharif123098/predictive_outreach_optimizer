# Project Charter: Next-Day Reminder-Call List for Appointment No-Shows

> Status: DRAFT written before modelling. Anything marked **[ASSUMPTION]** is my own choice, not a fact from the data. Anything marked **[MEASURE]** must be computed from the frozen dataset and filled in before modelling starts. Do not edit criteria after seeing results; if they change, record when and why in the Change Log at the bottom.

| Field | Content |
|---|---|
| **Business problem** | Booked appointments that patients do not attend leave empty slots and waste clinician time. Current no-show rate: **[MEASURE: share of rows with No-show = Yes in the frozen file]**. Staff cannot call every patient, so reminder calls need to go to the patients most likely to miss. |
| **Decision changed** | At 5 pm on the day before, the front-desk coordinator decides which of tomorrow's patients get a reminder call. Possible actions: call, do not call (optionally offer the slot to a waitlist if the patient cannot attend). |
| **Decision owner** | Clinic front-desk / scheduling coordinator. |
| **ML task** | Binary classification: probability that a booked appointment ends in a no-show. The output is a ranked call list, not a yes/no label. |
| **Target** | `No-show` column (Yes = patient did not attend), observed on the appointment day, after the decision point. |
| **Unit of prediction** | One appointment (one row, `AppointmentID`), scored at 5 pm the day before the appointment. |
| **Baseline to beat** | (1) Constant: predict the majority class. (2) Rule: call patients with the longest lead time between scheduling and appointment, flagging the same share as the model **[ASSUMPTION: this is a plausible existing practice; confirm and compute on the data]**. (3) Logistic regression on a few obvious features. All three are measured and logged as tracked runs: **[MEASURE]**. |
| **Offline criterion** | At a flag rate of 20% of next-day appointments **[ASSUMPTION: stands in for call capacity]**, precision must beat the lead-time rule's precision, and lift over the base no-show rate must be at least 1.25 **[ASSUMPTION: threshold chosen before seeing results]**. Also report PR-AUC with fold spread. Accuracy is not used. |
| **Business criterion** | Number of no-show patients reached per 100 calls, compared with the rule baseline. If a rupee/hour figure is wanted, it needs a clinic-specific slot cost I do not have: **[ASSUMPTION: cost of a missed slot is about 5x the cost of one call; replace with a justified value]**. |
| **Guardrails** | Report error rates by gender, age band and neighbourhood; the gap must be reported and discussed. Decide explicitly whether to exclude `Handcap`, `Neighbourhood` and `Gender` as inputs (the first is a disability-related flag): **[DECISION PENDING]**. Prediction must be explainable to staff in one sentence. API latency target: **[ASSUMPTION: p95 under 2 s]**. |
| **Kill criteria** | Recommend against deployment if: (a) the model does not beat the lead-time rule at the chosen flag rate; (b) lift is below 1.1 on the held-out later period; (c) a feature is shown to be unavailable at 5 pm the day before and removing it destroys performance. |
| **Known risks** | **Data is old and narrow:** it is a single public Kaggle snapshot covering about three months in one Brazilian city, so results may not transfer to any other clinic. **Leakage suspects:** `SMS_received` (is the SMS sent before 5 pm the day before? unresolved: **[DECISION PENDING]**, default is to exclude unless justified); any column written after the appointment. **Data quality:** check for negative lead times and invalid ages **[VERIFY against the file]**. **Short history:** per-patient history features will be thin, and a time-based split gives a short test window. **Famous dataset:** requires the new-question framing above and facilitator confirmation. |

## Split and feature rules (fixed now)

- **Split:** time-based on appointment date: train on earlier appointments, test on later ones, because the model scores future appointments. Test set is evaluated once.
- **Features:** only information available at 5 pm the day before. Patient history (prior no-shows) is built only from appointments dated before the one being scored.
- **Preprocessing:** all inside a scikit-learn Pipeline fitted on training data only.

## Data

Kaggle "Medical Appointment No Shows" (joniarroba/noshowappointments). Licence, download date, size and MD5: **[FILL in data/README.md after download]**. Column list in this charter is not yet verified against the file: **[VERIFY]**.

## Change Log

| Date | Field changed | Why |
|---|---|---|
| (none yet) | | |
