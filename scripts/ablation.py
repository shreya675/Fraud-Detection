"""Feature ablation: how much of the performance comes from the engineered
balance-reconciliation features?

Trains the same XGBoost configuration on (a) all 26 features and (b) only the
raw columns (type one-hot, amount, the four balances, time-of-day) and reports
validation / test PR-AUC for each.  Writes reports/ablation.json.

    python scripts/ablation.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import xgboost as xgb

from fraudguard.config import RANDOM_STATE, REPORTS_DIR, TARGET
from fraudguard.data import load_splits
from fraudguard.evaluate import compute_metrics, select_threshold
from fraudguard.features import build_preprocessor, get_feature_names

RAW_ONLY = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "hour_of_day",
    "day_of_week",
]
ENGINEERED_ONLY_DROP = [
    "error_balance_orig",
    "error_balance_dest",
    "amount_equals_orig_balance",
    "orig_emptied",
]


def fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, cols: list[str]) -> dict:
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)], verbose=False)
    p_va = model.predict_proba(X_va[cols])[:, 1]
    p_te = model.predict_proba(X_te[cols])[:, 1]
    thr = select_threshold(y_va, p_va)["threshold"]
    va = compute_metrics(y_va, p_va, thr)
    te = compute_metrics(y_te, p_te, thr)
    return {
        "n_features": len(cols),
        "best_iteration": int(model.best_iteration),
        "threshold": thr,
        "validation": {
            k: va[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fp", "fn")
        },
        "test": {k: te[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fp", "fn")},
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    train, val, test = load_splits()
    pre = build_preprocessor()
    X_tr = pre.fit_transform(train.drop(columns=[TARGET]))
    X_va = pre.transform(val.drop(columns=[TARGET]))
    X_te = pre.transform(test.drop(columns=[TARGET]))
    y_tr, y_va, y_te = (d[TARGET].to_numpy() for d in (train, val, test))
    all_cols = get_feature_names(pre)

    results = {
        "all_features": fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, all_cols),
        "without_reconciliation_features": fit_eval(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            y_te,
            [c for c in all_cols if c not in ENGINEERED_ONLY_DROP],
        ),
        "raw_columns_only": fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, RAW_ONLY),
    }
    results["_note"] = (
        "Same XGBoost configuration (scale_pos_weight=1, early stopping on validation PR-AUC); "
        "threshold = validation F1 optimum. Dropped in the second run: "
        + ", ".join(ENGINEERED_ONLY_DROP)
    )
    out = REPORTS_DIR / "ablation.json"
    out.write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(
            f"{k:34s} features={v['n_features']:2d}  test PR-AUC={v['test']['pr_auc']:.4f}  P={v['test']['precision']:.4f}  R={v['test']['recall']:.4f}  FP={v['test']['fp']}  FN={v['test']['fn']}"
        )
    print("wrote", out)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
