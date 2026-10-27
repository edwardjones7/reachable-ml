"""The four model pipelines. All share the same preprocessor, fit inside each CV fold."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data import SEED
from features import build_preprocessor


def make_models(features):
    estimators = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, early_stopping=True, random_state=SEED),
    }
    return {
        name: Pipeline([("prep", build_preprocessor(features)), ("clf", est)])
        for name, est in estimators.items()
    }
