# Milestone 1 — talk outline (≤30 min incl. Q&A)

Aim for about 12–15 minutes of talking, leaving the rest for questions. Open with the professor's
own objection and show it has been addressed. Every number below is in `results/`.

## 1. Problem description (~3 min)

- Reachable audits how a service business handles inbound leads and finds where they are lost.
- Feedback on the proposal: "integrating existing AI services is not sufficient." **Response:** the
  ML core contains no LLM and no API calls. It is a supervised model trained on labelled data and
  evaluated quantitatively.
- **ML problem:** binary classification. Given a lead and how it was handled, predict P(convert).
  - Output 1: a calibrated per-lead probability.
  - Output 2: a ranking of the *controllable* handling factors behind the predictions.
- **Data:** UCI Bank Marketing, 41,188 real contacts, 11.3% positive (Moro et al. 2014). It is the
  closest public analogue (channel, attempts, timing, prior contact, real outcomes). It is a proxy,
  and I'll come back to that.

## 2. Solution and implementation (~5 min)

- Features are split into **controllable** (channel, # attempts, recency, prior contacts, prior
  outcome, month, weekday) and **context** (demographics and macroeconomy, included so the
  controllable effects aren't confounded with them).
- `pdays = 999` is a sentinel for "never contacted". It becomes a flag plus the real gap.
- **Leakage decision (key slide):** `duration` is only known after the call ends. Conversion goes
  from 0.5% for the shortest calls to 33% for the longest. Model A drops it. Model B keeps it, only
  to measure the inflation. → `roc_a_vs_b.png`
- Four models on the same pipeline: majority baseline, logistic regression (balanced), random
  forest, HistGradientBoosting.
- Protocol: stratified 80/20 split, 5-fold CV on train for model selection, the test set used once,
  and the threshold chosen by F2 on out-of-fold training predictions (recall weighted).
  The sklearn `Pipeline`/`ColumnTransformer` refits preprocessing inside each fold, so nothing leaks.

## 3. Achieved so far (~5 min)

- All four models are trained on both variants. The table is in `results/results_table.md`.
- **Headline (Model A, test):** gradient boosting ROC-AUC 0.815, PR-AUC 0.495 against a no-skill
  0.113. Logistic regression scores 0.801 / 0.460.
- **Baseline slide:** the majority class gets 88.7% accuracy and 0 recall, which is why accuracy
  isn't the metric.
- **Leakage gap:** +0.140 ROC-AUC, +0.214 PR-AUC from `duration` alone.
- **Calibration:** balanced logistic regression is badly overconfident (Brier 0.162). Gradient
  boosting is already well calibrated (0.074), and isotonic calibration barely changes it.
  → `calibration_model_a.png`
- **Temporal split:** training on earlier campaigns and testing on later ones drops gradient
  boosting to ROC-AUC 0.641. The random split is optimistic because of macro drift.
- **Importance:** `nr.employed` (the economic cycle) dominates. Within the controllable set the
  order is prior contact/recency > channel > month > prior outcome ≈ number of attempts.
- **Honest framing:** the trees beat logistic regression by only about 0.02 PR-AUC, and the
  importances are correlational, not causal.

## Next steps (say these out loud)

- Tune the tree models (small grid inside CV). Report whether that beats logistic regression by
  more than the noise.
- Rerun model selection under the temporal split. Consider a model without macro features to see
  how much the controllable features carry on their own.
- Final writeup: transfer assumptions to home-services leads. Speed-to-lead is the missing feature.

## Likely questions — have answers ready

- *Why not accuracy?* Imbalance. See the baseline row.
- *Why permutation over impurity importance?* Impurity importance is biased toward high-cardinality
  features and is computed on training data. Permutation importance is measured on test.
- *Why F2?* Missing a convertible lead costs more than an extra follow-up.
- *Is GB really better than RF?* No. They are tied on CV PR-AUC (0.4676 vs 0.4675).
- *Does this transfer to home services?* Only the structure does. Say so and name the assumption.
