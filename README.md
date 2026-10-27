# Reachable — ML Core

Supervised model behind **Reachable**, a product that audits how a service business handles
inbound leads. CS07455 Machine Learning, Rowan University.

This repository contains **only** the trained, evaluated model. There are no LLM or API calls,
no transcription and no product UI. Everything below is trained from labelled data with
scikit-learn and evaluated quantitatively.

All numbers below come from `python src/run.py` with **random seed 42**. The full table is in
[`results/results_table.md`](results/results_table.md).

---

## 1. What is the specific ML problem? What exactly does the model predict?

**Binary classification.** Input: one inbound lead (contact attempt) plus how the business handled
it, meaning channel, number of attempts, timing and prior contact history. Output:

- **Per lead:** a calibrated probability that the lead converts (`y = 1`).
- **Global:** a ranking of which *controllable* handling factors the model's predictions
  depend on most, measured by permutation importance on held-out data.

The second output is what turns a score into a diagnostic: it says which part of lead handling
is most associated with lost conversions.

## 2. What data trains it?

**UCI Bank Marketing, dataset 222** (`bank-additional-full.csv`): 41,188 telemarketing contacts
from a Portuguese bank, May 2008 to November 2010, with real outcomes. **11.3% positive**,
so the classes are imbalanced.

> S. Moro, P. Cortez and P. Rita (2014). *A Data-Driven Approach to Predict the Success of Bank
> Telemarketing.* Decision Support Systems, 62, 22–31.

**Why this dataset:** each row is a contact attempt with a known conversion outcome. Its features
are channel, number of attempts, timing, prior contact and prior outcome, which map closely onto
inbound lead handling.

**It is a proxy, not the target population.** See [Limitations](#limitations).

```
python data/download.py      # fetches and extracts data/bank-additional-full.csv
```

## 3. What are the input features and target labels?

**Target:** `y` (`yes` → 1, `no` → 0).

Features are split into two groups, and the split is kept through the whole analysis.

| Group | Features |
|---|---|
| **Controllable** (what the business can change) | `contact` (channel), `campaign` (# contacts this campaign), `pdays` (days since last contact), `previous` (# prior contacts), `poutcome` (prior outcome), `month`, `day_of_week` |
| **Context** (not controllable, included to reduce confounding) | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| **Leaky** (Model B only) | `duration` |

**Preprocessing** (`src/features.py`). This is a single `ColumnTransformer` inside a `Pipeline`, so it
is refit inside every CV fold:

- `pdays == 999` is a sentinel meaning "never previously contacted" (96.3% of rows), not a
  999-day gap. It is split into a binary **`never_contacted`** flag plus **`pdays_contacted`**,
  the real gap. `pdays_contacted` is missing for never-contacted rows and median-imputed. For
  importance, the two columns are permuted together as one feature, `pdays`.
- Numeric features: median imputation and standardisation.
- Categorical features: one-hot encoding. `unknown` is kept as its own level.

### The leakage decision: `duration`

`duration` is call length. It is **only known after the call has ended**, so it cannot be used to
predict before contact. It is also nearly a proxy for the label. Every `duration == 0` call is a
non-conversion. Conversion rises from **0.5%** in the shortest-call quintile to **33%** in the
longest. The UCI documentation says it should be discarded for realistic prediction.

We therefore train two versions with identical pipelines:

- **Model A (realistic, headline):** `duration` dropped.
- **Model B (leaky):** `duration` included, reported **only** to measure the inflation.

| Gradient boosting, test set | ROC-AUC | PR-AUC |
|---|---|---|
| Model B (with `duration`) | 0.955 | 0.708 |
| Model A (realistic) | 0.815 | 0.495 |
| **Inflation from leakage** | **+0.140** | **+0.214** |

A reported AUC of 0.95 on this dataset almost always means `duration` was left in. That model
would look excellent in evaluation and fail in production, because the feature it relies on does
not exist when the prediction is needed.

## 4. What ML methods are implemented and compared?

Four models (`src/models.py`), all with the same preprocessing, features and splits:

1. **Majority-class baseline.** Always predicts "no".
2. **Logistic regression.** `class_weight='balanced'`, an interpretable linear baseline.
3. **Random forest.** 300 trees, `min_samples_leaf=10`.
4. **Gradient boosting.** `HistGradientBoostingClassifier`, learning rate 0.05, up to 300
   iterations with early stopping.

The best model is selected by **mean 5-fold CV PR-AUC on the training set**. It is then
calibrated with `CalibratedClassifierCV(method='isotonic', cv=5)`, fit on training data only.

## 5. How is the model quantitatively evaluated?

**Protocol**

- Stratified 80/20 train/test split (32,950 / 8,238).
- 5-fold stratified CV on the training set for model selection and threshold choice.
- The **test set is used once**, for the final numbers.
- **Temporal split** robustness check: train on the first 80% of rows (earlier campaigns), test
  on the last 20% (later campaigns).

**Operating threshold.** We pick the threshold that maximises **F2** on *out-of-fold training*
predictions, never on test. F2 weights recall twice as much as precision. For a lead-handling
audit, missing a lead that would have converted costs more than flagging one that would not.

### Results — Model A (realistic), random split

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | **Test PR-AUC** | Brier | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Majority baseline | 0.500 | 0.113 | 0.500 | 0.113 | 0.113 | 0.000 | 0.000 | 0.000 | **0.887** |
| Logistic regression | 0.790 ± 0.006 | 0.444 ± 0.009 | 0.801 | 0.460 | 0.162 | 0.410 | 0.625 | 0.496 | 0.857 |
| Random forest | 0.792 ± 0.003 | 0.468 ± 0.012 | 0.809 | 0.491 | 0.075 | 0.416 | 0.643 | 0.505 | 0.858 |
| Gradient boosting | 0.798 ± 0.004 | 0.468 ± 0.014 | 0.815 | 0.495 | 0.074 | 0.400 | 0.650 | 0.495 | 0.851 |
| Gradient boosting + isotonic | — | — | 0.816 | 0.496 | 0.074 | 0.370 | 0.670 | 0.476 | 0.834 |

### Results — Model B (leaky, `duration` included), random split

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.936 | 0.581 | 0.944 | 0.623 | 0.099 | 0.423 | 0.933 | 0.582 |
| Random forest | 0.943 | 0.650 | 0.952 | 0.699 | 0.055 | 0.449 | 0.925 | 0.604 |
| Gradient boosting | 0.947 | 0.658 | 0.955 | 0.708 | 0.052 | 0.479 | 0.923 | 0.630 |

### Reading the results

- **Accuracy is not the headline metric.** The majority baseline scores 88.7% accuracy and catches
  zero conversions. PR-AUC is the headline metric: its no-skill level is the 11.3% prevalence, and
  every learned model reaches roughly 4× that.
- **Tree ensembles beat logistic regression, but only modestly.** Gradient boosting improves CV
  PR-AUC by **+0.024**, about 2 fold standard deviations. Gradient boosting and random forest are
  **statistically tied** (0.4676 vs 0.4675). Gradient boosting is chosen only because it is
  marginally ahead and is the better-calibrated of the two on the test set. Logistic regression is
  a competitive, more interpretable alternative.
- **Calibration.** Logistic regression with `class_weight='balanced'` is badly overconfident
  (Brier 0.162). Its predicted 0.6 corresponds to an observed rate of about 0.2, because balanced
  weighting deliberately shifts probabilities upward. Gradient boosting is already close to the
  diagonal (Brier 0.0743). Isotonic calibration changes it very little (0.0742). We report that
  plainly and do not claim a large improvement. See `results/calibration_model_a.png`.
- **Confusion matrix** for calibrated gradient boosting at threshold 0.098: TP 622, FN 306,
  FP 1,061, TN 6,249. It catches 67% of converting leads, and about 1 in 2.7 flagged leads converts.

### Temporal robustness check

| Gradient boosting | Train positive rate | Test positive rate | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|
| Random split | 0.113 | 0.113 | 0.815 | 0.495 |
| Temporal split | 0.064 | 0.308 | 0.641 | 0.432 |

Performance drops sharply when the model is trained on early campaigns and tested on later ones.
Gradient boosting falls by **0.17 ROC-AUC**, and under this split it is the *worst* of the three
learners. Random forest scores 0.743 and logistic regression 0.699. The base rate also shifts from
6% to 31%. The macroeconomic features (`nr.employed`, `euribor3m`, `emp.var.rate`) drift across the
period, and a flexible model fits the earlier macro regime too closely. **The random-split numbers
are optimistic about deployment on future leads.** That matters for a product that would score
leads arriving after training.

### Feature importance

Permutation importance on the **test set**: the drop in PR-AUC when one feature is shuffled,
averaged over 10 repeats. Impurity importance is not used, because it is biased toward
high-cardinality features.

**Overall:** `nr.employed` dominates (0.204), well ahead of every other feature. Four of the five
macro indicators are in the top nine. Much of the model's signal is *when* the call happened in the
economic cycle, not *how* the lead was handled.

**Within the controllable set** (`results/permutation_importance_controllable.csv`):

| Rank | Feature | Drop in PR-AUC | Share of controllable importance |
|---|---|---|---|
| 1 | `pdays` (prior contact / recency) | 0.033 | 30% |
| 2 | `contact` (cellular vs landline) | 0.027 | 25% |
| 3 | `month` (timing) | 0.018 | 17% |
| 4 | `poutcome` (prior outcome) | 0.012 | 11% |
| 5 | `campaign` (number of attempts) | 0.012 | 11% |
| 6 | `day_of_week` | 0.005 | 5% |
| 7 | `previous` | 0.002 | 2% |

In raw rates, previously contacted leads convert at 64% versus 9% for never-contacted leads, and
cellular contacts at 14.7% versus 5.2% for landline.

## Limitations

- **Importance is correlational, not causal.** The model shows that certain handling patterns
  (prior contact, mobile channel, timing) *co-occur* with conversion. It cannot show that changing
  them *causes* more conversions. Leads that were contacted before, or that are reachable by
  mobile, likely differ in other ways. `month` is also confounded with the macro cycle. Any
  recommendation built on this table is a hypothesis to test, not a proven lever.
- **Proxy population.** This is outbound bank telemarketing (term deposits, 2008–2010 Portugal),
  not inbound home-services leads. The transfer assumption is that the *structure* of the problem
  carries over: contact channel, attempt count, recency and prior relationship relate to conversion
  in a similar way. The specific coefficients, base rate and macro effects will not transfer. Speed
  of first response, the factor most central to inbound lead handling, is **not present** in this
  dataset.
- **Distribution shift.** The temporal check shows performance degrades on later periods. A
  deployed model would need retraining on the target business's own data.
- **Modest model gains.** The best model beats logistic regression by about 0.02 PR-AUC. The value
  of this project is in the evaluation (leakage, calibration, temporal robustness), not in a large
  accuracy win.

## Reproduce

```
pip install -r requirements.txt
python data/download.py
python src/run.py            # ~2 minutes, writes everything in results/
```

## Repository layout

```
data/download.py        # fetches the UCI zip, extracts bank-additional-full.csv (data is gitignored)
src/data.py             # load, pdays sentinel handling, random + temporal splits
src/features.py         # controllable / context / leaky groups, ColumnTransformer
src/models.py           # the four pipelines
src/evaluate.py         # CV, threshold, metrics, permutation importance, plots
src/run.py              # end to end: Model A, Model B, calibration, temporal split
results/                # results_table.{md,csv}, summary.json, plots
notebooks/              # walkthrough notebook
```

Plots in `results/`: `roc_model_a.png`, `pr_model_a.png`, `roc_a_vs_b.png`,
`calibration_model_a.png` (all models, plus before/after calibration), `permutation_importance.png`,
`confusion_model_a.png`.
#   r e a c h a b l e - m l  
 