"""Feature groups (controllable vs context) and the shared preprocessing ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw feature -> columns it becomes after cleaning in data.load().
# Permutation importance is computed per raw feature, so derived columns are permuted together.
FEATURE_COLUMNS = {
    "pdays": ["never_contacted", "pdays_contacted"],
}

# What a business can change about how it handles a lead.
CONTROLLABLE = ["contact", "campaign", "pdays", "previous", "poutcome", "month", "day_of_week"]

# Not controllable; included so controllable effects are not confounded with them.
CONTEXT = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]

# Known only after the call ends. Model B only.
LEAKY = ["duration"]

CATEGORICAL = {
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
}


def columns_for(features):
    """Expand raw feature names into the cleaned column names."""
    return [c for f in features for c in FEATURE_COLUMNS.get(f, [f])]


def feature_set(include_duration):
    """Raw features used by Model A (realistic) or Model B (leaky)."""
    return CONTROLLABLE + CONTEXT + (LEAKY if include_duration else [])


def group_of(feature):
    if feature in LEAKY:
        return "leaky"
    return "controllable" if feature in CONTROLLABLE else "context"


def build_preprocessor(features):
    cols = columns_for(features)
    categorical = [c for c in cols if c in CATEGORICAL]
    numeric = [c for c in cols if c not in CATEGORICAL]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
