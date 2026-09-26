"""Build and *execute* notebooks/01_eda.ipynb without requiring Jupyter.

Each code cell is run in a shared namespace; stdout and matplotlib figures are
captured into real notebook outputs so the committed notebook shows genuine
numbers from the dataset.

    python scripts/build_eda_notebook.py
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# FraudGuard — Exploratory Data Analysis of PaySim

PaySim simulates mobile-money transactions over 30 days (744 hourly steps). Fraud is rare
(~0.13%) and only appears in `TRANSFER` and `CASH_OUT` transactions. This notebook looks at
the raw data, the class imbalance, the balance-reconciliation signal that makes fraud
detectable, and the time-aware split used for modelling.

> Run `python -m fraudguard.data` first so the processed splits exist.""",
    ),
    (
        "code",
        """import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from fraudguard.config import RAW_DATA_FILE, TARGET
from fraudguard.data import load_raw
from fraudguard.features import engineer_features

C = {"legit": "#2a78d6", "fraud": "#e34948", "grid": "#e6e5e1", "ink": "#52514e"}
df = load_raw(RAW_DATA_FILE)
print(df.shape)
df.head()""",
    ),
    ("markdown", "## Schema, missing values, duplicates"),
    (
        "code",
        """print(df.dtypes.to_string())
print("\\nmissing values per column:\\n", df.isna().sum().to_string())
print("\\nduplicate rows:", int(df.duplicated().sum()))
print("steps:", df.step.min(), "->", df.step.max())""",
    ),
    ("markdown", "## Class imbalance and fraud by transaction type"),
    (
        "code",
        """print("fraud rate: {:.4%}  ({:,} of {:,})".format(df[TARGET].mean(), int(df[TARGET].sum()), len(df)))
by_type = df.groupby("type", observed=True)[TARGET].agg(["count", "sum", "mean"]).rename(columns={"count": "rows", "sum": "fraud", "mean": "fraud_rate"})
by_type["fraud_rate"] = (by_type["fraud_rate"] * 100).round(3)
print(by_type.to_string())
print("\\nisFlaggedFraud (business rule) flags only", int(df.isFlaggedFraud.sum()), "rows")""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
by_type["rows"].sort_values().plot.barh(ax=ax, color=C["legit"])
ax.set_title("Transactions by type", loc="left", color=C["ink"]); ax.set_xlabel("rows")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color=C["grid"]); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()""",
    ),
    ("markdown", "## Amount distributions: fraud vs legitimate"),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(7, 3.8), dpi=120)
bins = np.logspace(0, 7.2, 60)
ax.hist(df.loc[df[TARGET] == 0, "amount"].clip(lower=1), bins=bins, color=C["legit"], alpha=0.8, label="legitimate")
ax.hist(df.loc[df[TARGET] == 1, "amount"].clip(lower=1), bins=bins, color=C["fraud"], alpha=0.9, label="fraud")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(frameon=False)
ax.set_title("Transaction amount (log-log)", loc="left", color=C["ink"]); ax.set_xlabel("amount")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(color=C["grid"]); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
print(df.groupby(TARGET)["amount"].describe().round(0).to_string())""",
    ),
    (
        "markdown",
        "## The balance-reconciliation signal\n\nFor a clean transaction `newbalanceOrig + amount == oldbalanceOrg` and `oldbalanceDest + amount == newbalanceDest`. Fraudulent PaySim transactions drain the origin account and the destination balance is frequently not updated, so the *destination error* is large.",
    ),
    (
        "code",
        """feat = engineer_features(df.sample(1_000_000, random_state=42))
cols = ["orig_emptied", "amount_equals_orig_balance", "dest_zero_before", "dest_zero_after", "orig_zero_before"]
summary = feat.groupby(TARGET)[cols].mean().T.rename(columns={0: "legit_rate", 1: "fraud_rate"}).round(3)
print(summary.to_string())
print("\\nabs destination error (median):")
print(feat.groupby(TARGET)["error_balance_dest"].apply(lambda s: s.abs().median()).round(2).to_string())""",
    ),
    (
        "markdown",
        "## Fraud over time and the time-aware split\n\nLegitimate traffic thins out after roughly step 400 while fraud continues at a steady rate, so the *last* 20% of steps (the test period) has a much higher fraud rate. This is genuine distribution shift between training and test and is one reason a chronological split is more honest than a random one.",
    ),
    (
        "code",
        """per_step = df.groupby("step")[TARGET].agg(["count", "sum"])
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=120)
ax.plot(per_step.index, per_step["count"], color=C["legit"], lw=1.5, label="all transactions / hour")
ax.plot(per_step.index, per_step["sum"] * 100, color=C["fraud"], lw=1.5, label="fraud / hour (x100)")
for x, lbl in [(281, "train | val"), (355, "val | test")]:
    ax.axvline(x, color=C["ink"], ls="--", lw=1); ax.text(x + 3, ax.get_ylim()[1] * 0.92, lbl, color=C["ink"], fontsize=8)
ax.set_title("Hourly volume over the simulation (split boundaries shown)", loc="left", color=C["ink"])
ax.set_xlabel("step (hour)"); ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(color=C["grid"]); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()""",
    ),
    (
        "code",
        """import json
from fraudguard.data import SUMMARY_FILE
summary = json.loads(SUMMARY_FILE.read_text())
pd.DataFrame(summary["splits"]).set_index("split")""",
    ),
    (
        "markdown",
        "## Takeaways\n\n* Clean dataset: no missing values, no duplicates, five transaction types.\n* Extreme imbalance (0.13% fraud) → PR-AUC, precision, recall and F1 are the right metrics; accuracy is meaningless.\n* Fraud only occurs in `TRANSFER` and `CASH_OUT`; fraudsters typically empty the origin account and the destination balance is not updated — captured by `orig_emptied`, `amount_equals_orig_balance` and `error_balance_dest`.\n* The chronological split exposes real drift: the test period's fraud rate is ~4x the training period's.",
    ),
]


def _fig_outputs() -> list[dict]:
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        outs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(buf.getvalue()).decode()},
            }
        )
        plt.close(fig)
    return outs


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    ns: dict = {}
    plt.show = lambda *a, **k: None  # figures are captured after each cell
    nb_cells = []
    n_exec = 0
    for kind, src in CELLS:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
            continue
        n_exec += 1
        stdout = io.StringIO()
        outputs: list[dict] = []
        with contextlib.redirect_stdout(stdout):
            lines = src.strip().split("\n")
            # emulate notebook behaviour: echo the value of a trailing expression
            try:
                body, last = "\n".join(lines[:-1]), lines[-1]
                exec(body, ns)
                try:
                    value = eval(last, ns)
                    if value is not None:
                        html = value._repr_html_() if hasattr(value, "_repr_html_") else None
                        data = {"text/plain": repr(value)}
                        if html:
                            data["text/html"] = html
                        outputs.append(
                            {
                                "output_type": "execute_result",
                                "execution_count": n_exec,
                                "metadata": {},
                                "data": data,
                            }
                        )
                except SyntaxError:
                    exec(last, ns)
            except Exception as exc:  # keep going but record the error
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": f"{type(exc).__name__}: {exc}\n",
                    }
                )
        text = stdout.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text})
        outputs.extend(_fig_outputs())
        nb_cells.append(
            {
                "cell_type": "code",
                "execution_count": n_exec,
                "metadata": {},
                "outputs": outputs,
                "source": src,
            }
        )
        print(f"cell {n_exec} ok ({len(outputs)} outputs)")

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
