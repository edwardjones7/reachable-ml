"""Metrics, threshold selection, grouped permutation importance and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from data import SEED
from features import columns_for, group_of

GROUP_COLORS = {"controllable": "#2a6fdb", "context": "#9aa3ad", "leaky": "#d1495b"}


def cross_validate_oof(model, X, y, n_splits=5, seed=SEED):
    """5-fold stratified CV on the training set. Returns out-of-fold probabilities
    and per-fold ROC-AUC / PR-AUC."""
    oof = np.zeros(len(y))
    folds = []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        folds.append((roc_auc_score(y.iloc[va], p), average_precision_score(y.iloc[va], p)))
    folds = np.array(folds)
    return oof, {
        "cv_roc_auc": folds[:, 0].mean(), "cv_roc_auc_sd": folds[:, 0].std(),
        "cv_pr_auc": folds[:, 1].mean(), "cv_pr_auc_sd": folds[:, 1].std(),
    }


def pick_threshold(y, p, beta=2.0):
    """Threshold maximising F-beta on out-of-fold training predictions.
    beta=2 weights recall twice as heavily as precision: a missed convertible lead
    costs more than an extra follow-up. Chosen on training data only, never on test."""
    if np.ptp(p) == 0:  # constant scores (majority baseline): no threshold to tune
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    fb = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    if len(thr) == 0 or np.all(fb == 0):
        return 0.5
    return float(thr[np.argmax(fb)])


def test_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def grouped_permutation_importance(model, X, y, features, n_repeats=10, seed=SEED):
    """Permutation importance on the test set, scored by drop in PR-AUC.
    Columns derived from one raw feature (pdays -> never_contacted + pdays_contacted)
    are shuffled together so the derived pieces can't cover for each other."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for f in features:
        cols = columns_for([f])
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[idx]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": f, "group": group_of(f),
                     "importance": np.mean(drops), "sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def plot_roc(curves, y, path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(curves, y, path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in curves.items():
        if np.ptp(p) == 0:  # a constant score's PR curve is just the prevalence line, drawn below
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="k", ls=":", lw=1,
               label=f"Majority baseline / prevalence (AP {y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, ylim=(0, 1.02))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(all_models, best_name, best_calibrated, y, path):
    """Left: reliability of every Model A learner. Right: best model before vs after calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfectly calibrated")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed conversion rate",
               xlim=(0, 1), ylim=(0, 1))
    for name, p in all_models.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[0].plot(mean, frac, "o-", ms=3,
                     label=f"{name} (Brier {brier_score_loss(y, p):.4f})")
    axes[0].set_title("Model A: calibration on test set")
    for label, p in [(f"{best_name} (before)", all_models[best_name]),
                     (f"{best_name} + isotonic (after)", best_calibrated)]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1].plot(mean, frac, "o-", ms=3,
                     label=f"{label} (Brier {brier_score_loss(y, p):.4f})")
    axes[1].set_title("Best model: before vs after calibration")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path, title):
    imp = imp.iloc[::-1]
    present = [g for g in GROUP_COLORS if g in set(imp["group"])]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["sd"],
            color=[GROUP_COLORS[g] for g in imp["group"]], ecolor="#444", capsize=2)
    ax.set(xlabel="Drop in test PR-AUC when feature is shuffled", title=title)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in present], present,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"], yticklabels=["no", "yes"],
           xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
