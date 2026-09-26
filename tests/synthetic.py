"""Small synthetic PaySim-like dataset for tests and CI.

Mimics the real dataset's schema and its dominant fraud pattern (the origin
account is emptied by a TRANSFER / CASH_OUT whose destination balances are
not updated), so a quick model trained on it behaves sensibly.  It is *not*
used for any reported metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import RAW_COLUMNS, TRANSACTION_TYPES


def make_synthetic_paysim(
    n_rows: int = 6000, fraud_rate: float = 0.03, n_steps: int = 120, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 20)
    n_legit = n_rows - n_fraud

    # --- legitimate ------------------------------------------------------- #
    types = rng.choice(TRANSACTION_TYPES, size=n_legit, p=[0.22, 0.35, 0.01, 0.34, 0.08])
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.2, size=n_legit), 2)
    old_org = np.round(rng.lognormal(mean=11, sigma=1.5, size=n_legit), 2)
    new_org = np.where(
        np.isin(types, ["CASH_IN"]), old_org + amount, np.clip(old_org - amount, 0, None)
    )
    old_dst = np.round(rng.lognormal(mean=12, sigma=1.5, size=n_legit), 2)
    new_dst = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]), old_dst + amount, old_dst)
    # merchants (PAYMENT) have unknown balances in PaySim
    is_payment = types == "PAYMENT"
    old_dst = np.where(is_payment, 0.0, old_dst)
    new_dst = np.where(is_payment, 0.0, new_dst)
    legit = pd.DataFrame(
        {
            "step": rng.integers(1, n_steps + 1, size=n_legit),
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{rng.integers(1e9)}" for _ in range(n_legit)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": np.round(new_org, 2),
            "nameDest": [("M" if p else "C") + str(rng.integers(1e9)) for p in is_payment],
            "oldbalanceDest": np.round(old_dst, 2),
            "newbalanceDest": np.round(new_dst, 2),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
    )

    # --- fraud: account drained, destination not updated ------------------ #
    ftypes = rng.choice(["TRANSFER", "CASH_OUT"], size=n_fraud)
    f_old_org = np.round(rng.lognormal(mean=12, sigma=1.2, size=n_fraud), 2)
    fraud = pd.DataFrame(
        {
            "step": rng.integers(1, n_steps + 1, size=n_fraud),
            "type": ftypes,
            "amount": f_old_org,
            "nameOrig": [f"C{rng.integers(1e9)}" for _ in range(n_fraud)],
            "oldbalanceOrg": f_old_org,
            "newbalanceOrig": 0.0,
            "nameDest": [f"C{rng.integers(1e9)}" for _ in range(n_fraud)],
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": (f_old_org > 200_000).astype(int),
        }
    )

    df = (
        pd.concat([legit, fraud], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    df["type"] = pd.Categorical(df["type"], categories=TRANSACTION_TYPES)
    df["isFraud"] = df["isFraud"].astype("int8")
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype("int8")
    return df[RAW_COLUMNS]
