"""End-to-end: train and evaluate Model A (no duration) and Model B (with duration),
random and temporal splits, calibration and permutation importance. Writes results/."""
import json
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import evaluate as ev
from data import SEED, load, random_split, temporal_split
from features import feature_set
from models import make_models

RESULTS = Path(__file__).resolve().parents[1] / "results"
TABLE_COLS = ["variant", "model", "cv_roc_auc", "cv_pr_auc", "roc_auc", "pr_auc", "brier",
              "threshold", "precision", "recall", "f1", "accuracy"]


def run_variant(label, include_duration, X_tr, X_te, y_tr, y_te):
    """CV on train for selection, then one fit on all of train and one look at test."""
    features = feature_set(include_duration)
    rows, test_probs, fitted = [], {}, {}
    for name, model in make_models(features).items():
        t = time.time()
        oof, cv = ev.cross_validate_oof(model, X_tr, y_tr)
        thr = ev.pick_threshold(y_tr, oof)
        fitted[name] = clone(model).fit(X_tr, y_tr)
        p = fitted[name].predict_proba(X_te)[:, 1]
        test_probs[name] = p
        rows.append({"variant": label, "model": name, **cv, **ev.test_metrics(y_te, p, thr)})
        print(f"  [{label}] {name:20s} CV PR-AUC {cv['cv_pr_auc']:.3f}  "
              f"test PR-AUC {rows[-1]['pr_auc']:.3f}  ({time.time() - t:.0f}s)")
    return pd.DataFrame(rows), test_probs, fitted, features


def calibrate(model, X_tr, y_tr, X_te, y_te):
    """Isotonic calibration, fit with internal 5-fold CV on train only.
    Its threshold is re-chosen on nested out-of-fold predictions, again train only."""
    cal = CalibratedClassifierCV(clone(model), method="isotonic", cv=5)
    oof = cross_val_predict(cal, X_tr, y_tr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
    thr = ev.pick_threshold(y_tr, oof)
    cal.fit(X_tr, y_tr)
    p = cal.predict_proba(X_te)[:, 1]
    return p, ev.test_metrics(y_te, p, thr)


def main():
    RESULTS.mkdir(exist_ok=True)
    X, y = load()
    print(f"Loaded {len(X):,} rows, positive rate {y.mean():.3f}, seed {SEED}")
    X_tr, X_te, y_tr, y_te = random_split(X, y)

    print("Model A: duration dropped")
    a_tab, a_probs, a_fit, a_feats = run_variant("A (no duration)", False, X_tr, X_te, y_tr, y_te)
    print("Model B: duration included (leakage demo)")
    b_tab, b_probs, _, _ = run_variant("B (duration, leaky)", True, X_tr, X_te, y_tr, y_te)

    # Model selection on CV PR-AUC only, never on test.
    learners = a_tab[a_tab["model"] != "Majority baseline"]
    best = learners.loc[learners["cv_pr_auc"].idxmax(), "model"]
    print(f"Best Model A by CV PR-AUC: {best}. Calibrating.")
    cal_p, cal_m = calibrate(a_fit[best], X_tr, y_tr, X_te, y_te)
    cal_row = {"variant": "A (no duration)", "model": f"{best} + isotonic", **cal_m}

    # Robustness: train on earlier campaigns, test on later ones.
    print("Temporal split (Model A)")
    Xt_tr, Xt_te, yt_tr, yt_te = temporal_split(X, y)
    t_tab, _, _, _ = run_variant("A, temporal split", False, Xt_tr, Xt_te, yt_tr, yt_te)

    table = pd.concat([a_tab, pd.DataFrame([cal_row]), b_tab, t_tab], ignore_index=True)
    table.to_csv(RESULTS / "results_table.csv", index=False)
    with open(RESULTS / "results_table.md", "w") as f:
        f.write(f"Random split: train {len(y_tr):,} / test {len(y_te):,} "
                f"(test positive rate {y_te.mean():.3f}). Temporal split: train {len(yt_tr):,} "
                f"/ test {len(yt_te):,} (train positive rate {yt_tr.mean():.3f}, "
                f"test positive rate {yt_te.mean():.3f}).\n\n")
        f.write(table[TABLE_COLS].to_markdown(index=False, floatfmt=".3f") + "\n")

    learner_probs = {k: v for k, v in a_probs.items() if k != "Majority baseline"}
    ev.plot_roc(a_probs, y_te, RESULTS / "roc_model_a.png", "ROC, Model A (no duration), test set")
    ev.plot_pr(a_probs, y_te, RESULTS / "pr_model_a.png", "Precision-recall, Model A, test set")
    ev.plot_roc({f"A: {k}": v for k, v in learner_probs.items()}
                | {f"B: {k}": v for k, v in b_probs.items() if k != "Majority baseline"},
                y_te, RESULTS / "roc_a_vs_b.png", "ROC, Model A vs Model B (leaky)")
    ev.plot_calibration(learner_probs, best, cal_p, y_te, RESULTS / "calibration_model_a.png")
    ev.plot_confusion(cal_m, RESULTS / "confusion_model_a.png",
                      f"{best} + isotonic\nthreshold {cal_m['threshold']:.3f}")

    print("Permutation importance (test set, 10 repeats)")
    imp = ev.grouped_permutation_importance(a_fit[best], X_te, y_te, a_feats)
    imp.to_csv(RESULTS / "permutation_importance.csv", index=False)
    ctrl = imp[imp["group"] == "controllable"].copy()
    pos = ctrl["importance"].clip(lower=0)
    ctrl["share_of_controllable"] = pos / pos.sum()
    ctrl.to_csv(RESULTS / "permutation_importance_controllable.csv", index=False)
    ev.plot_importance(imp, RESULTS / "permutation_importance.png",
                       f"Permutation importance, {best} (Model A)")

    a = a_tab.set_index("model")
    b = b_tab.set_index("model")
    t = t_tab.set_index("model")
    summary = {
        "seed": SEED,
        "best_model": best,
        "leakage_gap_roc_auc": b.loc[best, "roc_auc"] - a.loc[best, "roc_auc"],
        "leakage_gap_pr_auc": b.loc[best, "pr_auc"] - a.loc[best, "pr_auc"],
        "best_vs_logreg_cv_pr_auc": a.loc[best, "cv_pr_auc"] - a.loc["Logistic regression", "cv_pr_auc"],
        "temporal_minus_random_roc_auc": t.loc[best, "roc_auc"] - a.loc[best, "roc_auc"],
        "temporal_minus_random_pr_auc": t.loc[best, "pr_auc"] - a.loc[best, "pr_auc"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(table[TABLE_COLS].to_string(index=False, float_format="%.3f"))
    print(imp.to_string(index=False, float_format="%.4f"))
    print(ctrl.to_string(index=False, float_format="%.4f"))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
