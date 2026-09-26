"""Generate notebooks/02_colab_train.ipynb - a step-by-step Colab notebook that
runs the FraudGuard data preparation, feature engineering, model training and
SHAP explanation on Google Colab (free tier, ~12 GB RAM), then packages the
trained artifacts for download.

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_colab_train.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# FraudGuard — train on Google Colab

This notebook runs the FraudGuard pipeline end-to-end on Colab's free tier:

1. upload the project code and the PaySim zip
2. **dataset preparation** — clean + leakage-safe time-aware split
3. **feature engineering** — look at the engineered features on real rows
4. **model training & comparison** — Logistic Regression vs Random Forest vs XGBoost, threshold tuning
5. **explanations** — SHAP contributions + reason codes on real transactions
6. download `models/` + `reports/` to run the API and dashboard locally

**Runtime → Change runtime type → CPU** is enough (no GPU needed). Expect ~15–20 minutes total.""",
    ),
    (
        "markdown",
        "## 1a. Upload the project code\nRun the cell and choose `fraudguard_code.zip` from `C:\\Fraud Detection\\`.",
    ),
    (
        "code",
        """from google.colab import files
import os, zipfile, io

uploaded = files.upload()  # pick fraudguard_code.zip
name = next(iter(uploaded))
# extract under /content/project (NOT /content, which is on Colab's import path
# and would let the project folder shadow the installed `fraudguard` package)
with zipfile.ZipFile(io.BytesIO(uploaded[name])) as zf:
    zf.extractall("/content/project")
os.chdir("/content/project/fraudguard")
print("project files:", sum(len(f) for _, _, f in os.walk(".")))
!ls""",
    ),
    ("markdown", "## 1b. Install dependencies (~2 min)"),
    (
        "code",
        """%pip install -q -r requirements.txt
%pip install -q -e .
import sys, importlib
sys.path.insert(0, "/content/project/fraudguard/src")   # make sure the kernel sees the package
for m in [k for k in list(sys.modules) if k == "fraudguard" or k.startswith("fraudguard.")]:
    del sys.modules[m]
import xgboost, sklearn, shap, fraudguard
from fraudguard.data import load_splits  # sanity check that the real package is imported
print("OK  xgboost", xgboost.__version__, "| sklearn", sklearn.__version__, "| shap", shap.__version__)""",
    ),
    (
        "markdown",
        """## 1c. Get the PaySim dataset
**Option A (recommended):** upload `archive (1).zip` (186 MB) to your Google Drive, then run this cell and approve the Drive access prompt. Adjust `ZIP_PATH` if you put it in a sub-folder.

**Option B:** if you have a Kaggle API token, skip to the next cell.""",
    ),
    (
        "code",
        """from google.colab import drive
drive.mount("/content/drive")

ZIP_PATH = "/content/drive/MyDrive/archive (1).zip"   # <- change if needed

import zipfile, os
os.makedirs("data/raw", exist_ok=True)
with zipfile.ZipFile(ZIP_PATH) as zf:
    zf.extractall("data/raw")
!ls -la data/raw""",
    ),
    (
        "code",
        """# Option B - Kaggle API (only if you did NOT use Option A)
# from google.colab import files; files.upload()      # upload kaggle.json
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# %pip install -q kaggle
# !python scripts/download_data.py""",
    ),
    (
        "markdown",
        """## 2. Dataset preparation
`fraudguard.data` loads the CSV with explicit dtypes, removes duplicates / missing / invalid rows (PaySim is clean, so expect zeros), drops the ID columns and the post-hoc `isFlaggedFraud` rule, and splits **chronologically on `step`**: first 60 % of rows → train, next 20 % → validation, last 20 % → untouched test. No hour is shared between splits, so the model is always evaluated on the future.""",
    ),
    ("code", "!python -m fraudguard.data"),
    (
        "code",
        """import json, pandas as pd
summary = json.load(open("reports/data_summary.json"))
display(pd.DataFrame(summary["splits"]).set_index("split"))
print("cleaning:", summary["cleaning"])
print("fraud by type:", summary["dataset"]["fraud_by_type"])""",
    ),
    (
        "markdown",
        """Notice the test period's fraud rate is ~4× the training period's: legitimate traffic thins out late in the simulation while fraud continues at a steady rate. That is real distribution shift, and we keep it.

## 3. Feature engineering
`features.engineer_features` adds 21 row-local features (no cross-row aggregates → no leakage). The important ones are the **balance-reconciliation errors** and **emptied-account flags**. Let's look at them on real fraud vs legitimate rows.""",
    ),
    (
        "code",
        """from fraudguard.data import load_splits
from fraudguard.features import engineer_features, build_preprocessor, get_feature_names

train, val, test = load_splits()
cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "error_balance_orig", "error_balance_dest", "orig_emptied", "amount_equals_orig_balance", "isFraud"]
sample = pd.concat([train[train.isFraud == 1].head(4), train[train.isFraud == 0].sample(4, random_state=1)])
display(engineer_features(sample)[cols])

feat = engineer_features(train.sample(500_000, random_state=0))
flags = ["orig_emptied", "amount_equals_orig_balance", "dest_zero_before", "orig_zero_before"]
display(feat.groupby("isFraud")[flags].mean().T.rename(columns={0: "legit rate", 1: "fraud rate"}).round(3))

pre = build_preprocessor().fit(train.drop(columns=["isFraud"]))
print(len(get_feature_names(pre)), "model features:", get_feature_names(pre))""",
    ),
    (
        "markdown",
        """`amount_equals_orig_balance` is true for ~98 % of frauds and ~0 % of legitimate transactions — the fraudster drains the account. This single feature is why every model in the next step scores near the ceiling: PaySim is a *simulator* with a mechanical fraud signature. Keep that in mind when reading the numbers (it goes in the README's limitations).

## 4. Train, compare, select (~45–60 min on Colab CPU)
`fraudguard.train` fits the preprocessor on train only, then gives every model family the same treatment: a small **hyper-parameter grid** (LR: 3 configs · RF: 6 · XGBoost: 12) is scored on validation PR-AUC using a 1M-row subsample of train, the best config is refit on all 3.8M rows, each model's threshold is tuned on validation (max-F1, plateau midpoint), and the selected model is reported **once** on the untouched test set.

Add `--deploy xgboost` (or `random_forest`) to deploy a model by explicit decision; the automatic choice is still recorded in the reports.""",
    ),
    (
        "code",
        "!python -m fraudguard.train 2>&1 | grep -v Warning   # add --deploy xgboost to override the automatic choice",
    ),
    (
        "code",
        """comp = json.load(open("reports/model_comparison.json"))
rows = []
for name, b in comp["models"].items():
    for split in ("validation_at_own_threshold", "test_at_own_threshold"):
        m = b[split]
        rows.append({"model": name, "split": split.split("_")[0], "thr": m["threshold"], "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "FP": m["fp"], "FN": m["fn"]})
display(pd.DataFrame(rows).set_index(["model", "split"]).round(6))
print("selected:", comp["selected_model"])
print("rule:", comp["selection_rule"])
for name, b in comp["models"].items():
    print(f"\n{name}: best params = {b['training']['best_params']}")
    display(pd.json_normalize(b["training"]["search_trials"]).rename(columns=lambda c: c.replace("params.", "")))
print({n: b["training"]["inference_ms_per_100k_rows"] for n, b in comp["models"].items()}, "ms per 100k rows")""",
    ),
    (
        "code",
        """from IPython.display import Image, display as show
for f in ["pr_curve_test.png", "roc_curve_test.png", "threshold_analysis.png", "confusion_matrix_test.png", "shap_feature_importance.png"]:
    p = f"reports/figures/{f}"
    if os.path.exists(p):
        show(Image(p, width=620))""",
    ),
    (
        "markdown",
        """## 5. Explanations on real transactions
The deployed model explains each score with SHAP (XGBoost's built-in TreeSHAP, or `shap.TreeExplainer` for a random forest) and turns the top contributions into short reason codes.""",
    ),
    (
        "code",
        """from fraudguard.predictor import FraudPredictor
p = FraudPredictor()
print("deployed model:", p.model_type, "| threshold:", p.threshold, "| SHAP:", p.supports_shap)

fraud_row = test[test.isFraud == 1].iloc[0].drop("isFraud").to_dict()
legit_row = test[test.isFraud == 0].iloc[0].drop("isFraud").to_dict()
for label, tx in [("REAL FRAUD", fraud_row), ("REAL LEGIT", legit_row)]:
    r = p.predict_one(tx, top_n=5)
    print(f"\\n{label}: p={r.fraud_probability:.4f} -> {'FRAUD' if r.is_fraud else 'legit'} ({r.risk_level})")
    if r.explanation:
        for reason in r.explanation.reasons: print("  •", reason)
        display(pd.DataFrame(r.explanation.top_features))""",
    ),
    (
        "code",
        """# Batch scoring on a slice of the untouched test set (with reasons)
scored = p.predict_frame(test.head(2000), explain=True, top_n=3)
print("flag rate:", scored.is_fraud.mean().round(4), "| actual fraud rate:", test.head(2000).isFraud.mean().round(4))
display(scored[scored.is_fraud][["type", "amount", "fraud_probability", "risk_level", "reasons"]].head())""",
    ),
    (
        "markdown",
        "## 6. (Optional) feature ablation — how much do the engineered features matter? (~5 min)",
    ),
    ("code", "!python scripts/ablation.py"),
    (
        "markdown",
        """## 7. Download the artifacts
Unzip `fraudguard_artifacts.zip` into `C:\\Fraud Detection\\fraudguard\\` (it contains `models/`, `reports/`, `data/samples/`). The API, dashboard, tests and Docker steps run locally from these files.""",
    ),
    (
        "code",
        """!zip -qr /content/fraudguard_artifacts.zip models reports data/samples
from google.colab import files
files.download("/content/fraudguard_artifacts.zip")""",
    ),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "name": "FraudGuard - train on Colab"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
