
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from itertools import product
np.random.seed(42)

def best_f1_threshold(y_true, scores, n_thresholds=200):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    f1s = []

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred))

    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]

def eval_isolation_forest(params, X_train, X_val, y_val):
    model = IsolationForest(
        **params,
        contamination="auto",
    )

    model.fit(X_train)

    # decision_function: higher = more normal → invert
    scores = -model.decision_function(X_val)

    auc = roc_auc_score(y_val, scores)
    threshold, f1 = best_f1_threshold(y_val, scores)

    return {
        "params": params,
        "auc": auc,
        "f1": f1,
        "threshold": threshold
    }

def randomized_search_iforest(
    param_distributions,
    X_train,
    X_val,
    y_val,
    n_iter=50,
):
    results = []

    sampler = ParameterSampler(
        param_distributions,
        n_iter=n_iter,
    )

    for params in sampler:
        res = eval_isolation_forest(params, X_train, X_val, y_val)
        results.append(res)

    return sorted(results, key=lambda x: (x["f1"], x["auc"]), reverse=True)

def eval_lof(params, X_train, X_val, y_val):
    model = LocalOutlierFactor(
        **params,
        novelty=True,
        metric='minkowski'
    )

    model.fit(X_train)

    # negative_outlier_factor_: lower = more anomalous → invert
    scores = -model.decision_function(X_val)

    auc = roc_auc_score(y_val, scores)
    threshold, f1 = best_f1_threshold(y_val, scores)

    return {
        "params": params,
        "auc": auc,
        "f1": f1,
        "threshold": threshold
    }

def grid_search_lof(param_grid, X_train, X_val, y_val):
    results = []

    keys = param_grid.keys()
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        res = eval_lof(params, X_train, X_val, y_val)
        results.append(res)

    return sorted(results, key=lambda x: (x["f1"], x["auc"]), reverse=True)
