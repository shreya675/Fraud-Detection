"""FraudGuard dashboard.

    streamlit run dashboard/app.py

Pages: Overview · Score a transaction · Batch scoring · Model performance ·
Prediction history.  Talks to the FastAPI service when it is running,
otherwise scores in-process.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run dashboard/app.py` from the repo root without installing the package
_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dashboard.utils import (  # noqa: E402
    EXAMPLE_TRANSACTIONS,
    Backend,
    comparison_table,
    importance_figure,
    load_reports,
    metric_bars_figure,
    probability_hist_figure,
    read_uploaded_csv,
    scored_to_csv_bytes,
    shap_bar_figure,
    template_csv_bytes,
    threshold_figure,
)
from fraudguard.config import INPUT_COLUMNS, TRANSACTION_TYPES  # noqa: E402

try:
    st.set_page_config(
        page_title="FraudGuard",
        page_icon=":material/shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:  # older Streamlit without material icons
    st.set_page_config(page_title="FraudGuard", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
#MainMenu, footer, header [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
h1 {font-size: 1.75rem !important; margin-bottom: 0.2rem !important;}
h2 {font-size: 1.2rem !important; margin-top: 1.4rem !important;}
h3 {font-size: 1.02rem !important;}
p, li {line-height: 1.55;}

.fg-brand {font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em; margin: 0;}
.fg-brand small {display:block; font-weight: 400; font-size: 0.78rem; color: #6b6a66; margin-top: 2px;}
.fg-topline {color: #6b6a66; font-size: 0.9rem; margin-bottom: 1.2rem;}
.fg-chip {display:inline-block; padding: 2px 9px; border-radius: 999px; background:#eeede8; color:#3a3936;
          font-size: 0.78rem; margin-right: 6px; border: 1px solid #e2e1db;}
.fg-chip.ok {background:#e6f4ec; color:#1c6b3c; border-color:#cfe8d8;}
.fg-chip.warn {background:#fdf1e3; color:#8a4b00; border-color:#f3dfc2;}

.fg-cards {display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 0.6rem 0 1rem 0;}
.fg-card {background:#ffffff; border:1px solid #e6e5e1; border-radius: 10px; padding: 14px 16px;}
.fg-card .l {font-size: 0.78rem; color:#6b6a66; text-transform: uppercase; letter-spacing: 0.04em;}
.fg-card .v {font-size: 1.55rem; font-weight: 650; margin-top: 2px; line-height: 1.1;}
.fg-card .n {font-size: 0.8rem; color:#6b6a66; margin-top: 4px;}

.fg-verdict {border-radius: 12px; padding: 18px 20px; border: 1px solid #e6e5e1; background:#ffffff;}
.fg-verdict.fraud {border-color:#f0c3c2; background:#fdf3f3;}
.fg-verdict.ok {border-color:#cfe8d8; background:#f2faf5;}
.fg-verdict .label {font-size: 0.8rem; color:#6b6a66; text-transform: uppercase; letter-spacing: 0.04em;}
.fg-verdict .big {font-size: 2.4rem; font-weight: 700; line-height: 1.05; margin: 2px 0 6px 0;}
.fg-verdict.fraud .big {color:#b3261e;}
.fg-verdict.ok .big {color:#1c6b3c;}
.fg-verdict .sub {font-size: 0.95rem; color:#3a3936;}
.fg-verdict ul {margin: 8px 0 0 0; padding-left: 1.1rem;}
.fg-verdict li {margin: 2px 0;}

.fg-muted {color:#6b6a66; font-size: 0.88rem;}
.fg-empty {border:1px dashed #d9d8d2; border-radius: 12px; padding: 28px 20px; color:#6b6a66; text-align:center;}
section[data-testid="stSidebar"] {border-right: 1px solid #e6e5e1;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
div[data-testid="stForm"] {border: 1px solid #e6e5e1; border-radius: 12px; padding: 14px 16px 6px 16px; background:#ffffff;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def cards(items: list[tuple[str, str, str]]) -> None:
    """Row of metric cards: (label, value, note)."""
    html = "".join(
        f'<div class="fg-card"><div class="l">{lbl}</div><div class="v">{val}</div><div class="n">{note}</div></div>'
        for lbl, val, note in items
    )
    st.markdown(f'<div class="fg-cards">{html}</div>', unsafe_allow_html=True)


def pct(x: float | None, digits: int = 2) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def plot(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df, height: int | None = None) -> None:
    kwargs = {"height": height, "hide_index": True} if height else {"hide_index": True}
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def image(path: Path, caption: str) -> None:
    try:
        st.image(str(path), caption=caption, width="stretch")
    except TypeError:
        st.image(str(path), caption=caption, use_container_width=True)


def local_time(series: pd.Series) -> pd.Series:
    """UTC ISO timestamps -> local wall-clock time, second precision."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    local_tz = datetime.now().astimezone().tzinfo  # the machine's own timezone
    return ts.dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d %H:%M:%S")


def join_reasons(series: pd.Series) -> pd.Series:
    return series.apply(lambda r: "; ".join(r) if isinstance(r, (list, tuple)) else r)


@st.cache_resource(show_spinner="Connecting to the scoring service…")
def get_backend() -> Backend:
    return Backend(prefer_api=True)


@st.cache_data(ttl=60)
def get_model_info() -> dict:
    return get_backend().model_info()


@st.cache_data(ttl=300)
def get_reports() -> dict:
    return load_reports()


# --------------------------------------------------------------------------- #
# Load model + sidebar
# --------------------------------------------------------------------------- #
try:
    backend = get_backend()
    info = get_model_info()
    model_ok = True
except Exception as exc:  # model not trained yet
    backend, info, model_ok = None, {}, False
    load_error = str(exc)

trained_threshold = float(info.get("threshold", {}).get("threshold", 0.5)) if model_ok else 0.5
test_m = (info.get("test_metrics") or {}) if model_ok else {}

with st.sidebar:
    st.markdown(
        '<p class="fg-brand">FraudGuard<small>Transaction fraud scoring</small></p>',
        unsafe_allow_html=True,
    )
    st.write("")
    page = st.radio(
        "Pages",
        [
            "Overview",
            "Score a transaction",
            "Batch scoring",
            "Model performance",
            "Prediction history",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Decision threshold**")
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = trained_threshold
    threshold = st.slider(
        "Decision threshold",
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        key="threshold",
        label_visibility="collapsed",
    )
    if abs(threshold - trained_threshold) > 1e-9:
        st.markdown(
            f'<span class="fg-muted">Trained value is {trained_threshold:.2f}. Lower catches more fraud '
            f"but raises false alarms; higher does the opposite.</span>",
            unsafe_allow_html=True,
        )
        st.button(
            "Reset to trained value",
            on_click=lambda: st.session_state.update(threshold=trained_threshold),
        )
    else:
        st.markdown(
            '<span class="fg-muted">Tuned on validation data (max F1). Transactions at or above it are flagged.</span>',
            unsafe_allow_html=True,
        )
    st.divider()
    if model_ok:
        mode = "API" if backend.mode == "api" else "in-process"
        st.markdown(
            f'<span class="fg-muted">Model: {info.get("model_type")}<br>Scoring: {mode}</span>',
            unsafe_allow_html=True,
        )

if not model_ok:
    st.title("FraudGuard")
    st.error(
        f"No trained model found ({load_error}). Run `python -m fraudguard.data` and then "
        "`python -m fraudguard.train` first, or copy the `models/` folder from a training run."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    st.title("FraudGuard")
    st.markdown(
        '<div class="fg-topline">Scores PaySim mobile-money transactions for fraud, explains each score, '
        "and keeps a record of everything it has scored.</div>",
        unsafe_allow_html=True,
    )
    chips = (
        f'<span class="fg-chip ok">{info.get("model_type")}</span>'
        f'<span class="fg-chip">threshold {trained_threshold:.2f}</span>'
        f'<span class="fg-chip">{info.get("n_features")} features</span>'
        f'<span class="fg-chip">{"API connected" if backend.mode == "api" else "in-process scoring"}</span>'
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.subheader("Held-out test performance")
    fn, tp, fp = test_m.get("fn", 0), test_m.get("tp", 0), test_m.get("fp", 0)
    positives, n = test_m.get("positives", 0), test_m.get("n", 0)
    cards(
        [
            ("Recall", pct(test_m.get("recall")), f"{fn:,} of {positives:,} frauds missed"),
            (
                "Precision",
                pct(test_m.get("precision")),
                f"{fp:,} false alarms in {n:,} transactions",
            ),
            (
                "PR-AUC",
                f"{test_m.get('pr_auc', float('nan')):.4f}",
                "precision/recall across all thresholds",
            ),
            (
                "F1",
                f"{test_m.get('f1', float('nan')):.4f}",
                f"at threshold {trained_threshold:.2f}",
            ),
        ]
    )
    st.markdown(
        f'<span class="fg-muted">Test set = the last 20% of the timeline ({n:,} transactions), never used for '
        "training, model selection or threshold tuning.</span>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What it does")
        st.write(
            "Each transaction is turned into 26 features — mostly checks on whether the balances "
            "add up (does `oldbalanceOrg − amount` equal `newbalanceOrig`?), whether the origin "
            "account was drained, and ratios of amount to balance — and scored by a gradient-boosted "
            "tree model. The score is compared with the decision threshold, and the top SHAP "
            "contributions are translated into short reasons an analyst can act on."
        )
        st.write(
            "Use **Score a transaction** to try one case, **Batch scoring** to upload a CSV, and "
            "**Model performance** to see how the deployed model compares with the alternatives."
        )
    with right:
        st.subheader("How the model was chosen")
        reports = get_reports()
        comp = reports.get("comparison")
        if comp:
            tbl = comparison_table(comp, "validation_at_0.5")
            rf = comp["models"].get("random_forest", {}).get("test_at_own_threshold", {})
            xg = comp["models"].get("xgboost", {}).get("test_at_own_threshold", {})
            st.write(
                "Logistic Regression, Random Forest and XGBoost were each tuned with a small grid and "
                "compared on validation PR-AUC using a chronological split (train on the first 60% of "
                "hours, validate on the next 20%). "
                f"Random Forest and XGBoost tied (PR-AUC {tbl.loc['random_forest', 'pr_auc']:.4f} vs "
                f"{tbl.loc['xgboost', 'pr_auc']:.4f}); XGBoost was deployed because it provides exact "
                "SHAP explanations natively. On the test set Random Forest missed "
                f"{rf.get('fn', '–')} frauds and XGBoost {xg.get('fn', '–')}, with no false alarms from either."
            )
        st.write(
            "PaySim is a simulator, and its fraud follows a mechanical pattern (the account is "
            "emptied and the destination balance is not updated), which is why every model scores "
            "close to the ceiling. Real-world fraud is noisier; treat these numbers as an upper bound."
        )

    with st.expander("Model metadata"):
        st.json(
            {
                k: info[k]
                for k in (
                    "model_type",
                    "trained_at_utc",
                    "n_features",
                    "threshold",
                    "validation_metrics",
                    "test_metrics",
                    "split_strategy",
                    "environment",
                )
                if k in info
            }
        )


# --------------------------------------------------------------------------- #
# Score a transaction
# --------------------------------------------------------------------------- #
elif page == "Score a transaction":
    st.title("Score a transaction")
    st.markdown(
        '<div class="fg-topline">Enter the transaction as the payment system saw it, including the '
        "balances before and after. Balances are the main signal.</div>",
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([0.95, 1.05], gap="large")

    with form_col:
        preset = st.selectbox(
            "Start from an example",
            ["Custom"] + list(EXAMPLE_TRANSACTIONS),
            index=1,
        )
        base = EXAMPLE_TRANSACTIONS.get(
            preset, EXAMPLE_TRANSACTIONS["Suspicious TRANSFER (account emptied)"]
        )
        with st.form("single"):
            st.markdown("**Transaction**")
            c1, c2 = st.columns(2)
            ttype = c1.selectbox(
                "Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index(base["type"])
            )
            amount = c2.number_input(
                "Amount", min_value=0.0, value=float(base["amount"]), format="%.2f"
            )
            step = st.number_input(
                "Hour of simulation (step)",
                min_value=0,
                value=int(base["step"]),
                help="Only the hour of day and day of week are used as features.",
            )
            st.markdown("**Origin account**")
            c3, c4 = st.columns(2)
            old_org = c3.number_input(
                "Balance before",
                min_value=0.0,
                value=float(base["oldbalanceOrg"]),
                format="%.2f",
                key="oo",
            )
            new_org = c4.number_input(
                "Balance after",
                min_value=0.0,
                value=float(base["newbalanceOrig"]),
                format="%.2f",
                key="no",
            )
            st.markdown("**Destination account**")
            c5, c6 = st.columns(2)
            old_dst = c5.number_input(
                "Balance before",
                min_value=0.0,
                value=float(base["oldbalanceDest"]),
                format="%.2f",
                key="od",
            )
            new_dst = c6.number_input(
                "Balance after",
                min_value=0.0,
                value=float(base["newbalanceDest"]),
                format="%.2f",
                key="nd",
            )
            submitted = st.form_submit_button("Score", type="primary")

    with result_col:
        if not submitted and "last_single" not in st.session_state:
            st.markdown(
                '<div class="fg-empty">The score, decision and the reasons behind it will appear here.</div>',
                unsafe_allow_html=True,
            )
        else:
            if submitted:
                tx = {
                    "step": int(step),
                    "type": ttype,
                    "amount": amount,
                    "oldbalanceOrg": old_org,
                    "newbalanceOrig": new_org,
                    "oldbalanceDest": old_dst,
                    "newbalanceDest": new_dst,
                }
                try:
                    st.session_state["last_single"] = backend.predict_one(
                        tx, threshold=threshold, top_n=8
                    )
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.stop()
            res = st.session_state["last_single"]
            p = res["fraud_probability"]
            flagged = res["is_fraud"]
            expl = res.get("explanation") or {}
            reasons = expl.get("reasons") or []
            reasons_html = (
                "<ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"
                if reasons
                else '<div class="sub">No single dominant risk factor — see the contributions below.</div>'
                if flagged
                else '<div class="sub">Nothing in this transaction pushes the score up materially.</div>'
            )
            st.markdown(
                f'<div class="fg-verdict {"fraud" if flagged else "ok"}">'
                f'<div class="label">Decision at threshold {res["threshold"]:.2f}</div>'
                f'<div class="big">{"Flag as fraud" if flagged else "Looks legitimate"}</div>'
                f'<div class="sub">Fraud probability <b>{p:.4f}</b> · risk level {res["risk_level"].lower()}</div>'
                f"{reasons_html}</div>",
                unsafe_allow_html=True,
            )
            if expl.get("top_features"):
                plot(shap_bar_figure(expl["top_features"], title="What drove this score"))
                st.markdown(
                    '<span class="fg-muted">SHAP contributions in log-odds. Red bars raise the fraud '
                    "score, blue bars lower it; together with the base value they add up to the "
                    "model output.</span>",
                    unsafe_allow_html=True,
                )
            with st.expander("Raw response"):
                st.json(res)


# --------------------------------------------------------------------------- #
# Batch scoring
# --------------------------------------------------------------------------- #
elif page == "Batch scoring":
    st.title("Batch scoring")
    st.markdown(
        '<div class="fg-topline">Upload a CSV of transactions. Required columns: '
        + ", ".join(f"<code>{c}</code>" for c in INPUT_COLUMNS)
        + ". Any other columns are passed through unchanged.</div>",
        unsafe_allow_html=True,
    )

    up_col, side_col = st.columns([1.3, 0.7], gap="large")
    with up_col:
        uploaded = st.file_uploader("Transactions CSV", type=["csv"], label_visibility="collapsed")
    with side_col:
        st.download_button(
            "Download a template CSV", template_csv_bytes(), "fraudguard_template.csv", "text/csv"
        )
        st.markdown(
            '<span class="fg-muted">Tip: <code>data/samples/sample_transactions.csv</code> has 525 rows '
            "from the held-out test set, including ~25 known frauds.</span>",
            unsafe_allow_html=True,
        )

    if uploaded is not None:
        try:
            df = read_uploaded_csv(uploaded)
        except Exception as e:
            st.error(str(e))
            st.stop()
        c1, c2 = st.columns([0.25, 0.75])
        run = c1.button(f"Score {len(df):,} rows", type="primary")
        if run:
            with st.spinner("Scoring…"):
                try:
                    st.session_state["last_batch"] = backend.predict_batch(df, threshold=threshold)
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.stop()

    scored = st.session_state.get("last_batch")
    if scored is not None:
        used_thr = float(scored["threshold"].iloc[0])
        flagged = int(scored["is_fraud"].sum())
        items = [
            ("Scored", f"{len(scored):,}", "rows"),
            (
                "Flagged",
                f"{flagged:,}",
                f"{pct(flagged / len(scored))} of rows at threshold {used_thr:.2f}",
            ),
        ]
        if "isFraud" in scored.columns:
            y = scored["isFraud"].astype(int)
            tp = int(((y == 1) & scored["is_fraud"]).sum())
            fp = int(((y == 0) & scored["is_fraud"]).sum())
            fn = int(((y == 1) & ~scored["is_fraud"]).sum())
            items += [
                ("Caught", f"{tp} / {tp + fn}", "of the labelled frauds (isFraud column found)"),
                ("False alarms", f"{fp}", f"precision {tp / max(tp + fp, 1):.3f}"),
            ]
        cards(items)

        st.subheader("Flagged transactions")
        flagged_rows = scored[scored["is_fraud"]].sort_values("fraud_probability", ascending=False)
        if flagged_rows.empty:
            st.markdown(
                '<div class="fg-empty">Nothing was flagged at this threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            show = flagged_rows.copy()
            if "reasons" in show.columns:
                show["reasons"] = join_reasons(show["reasons"])
            cols = [
                c
                for c in [
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "fraud_probability",
                    "risk_level",
                    "reasons",
                ]
                if c in show.columns
            ]
            table(show[cols], height=min(420, 60 + 35 * len(show)))

        with st.expander("All rows and score distribution"):
            plot(probability_hist_figure(scored, used_thr))
            full = scored.sort_values("fraud_probability", ascending=False).copy()
            if "reasons" in full.columns:
                full["reasons"] = join_reasons(full["reasons"])
            table(full, height=420)

        st.download_button(
            "Download scored CSV",
            scored_to_csv_bytes(scored),
            "fraudguard_scored.csv",
            "text/csv",
            type="primary",
        )


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
elif page == "Model performance":
    st.title("Model performance")
    reports = get_reports()
    comp = reports["comparison"]
    if not comp:
        st.warning("No training reports found. Run `python -m fraudguard.train`.")
        st.stop()

    st.markdown(
        f'<div class="fg-topline">Deployed model: <b>{comp["selected_model"]}</b>. Models were compared '
        "and thresholds tuned on the validation split; the test numbers below were computed once, "
        "after every decision was made.</div>",
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "Threshold", "Feature importance", "Figures"])

    with tab1:
        tbl = comparison_table(comp, "test_at_own_threshold")
        table(
            tbl.style.format(
                {
                    "threshold": "{:.2f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "pr_auc": "{:.5f}",
                    "roc_auc": "{:.5f}",
                }
            )
        )
        plot(metric_bars_figure(tbl))
        with st.expander("Validation-set comparison (what the selection was based on)"):
            table(comparison_table(comp, "validation_at_own_threshold"))
            st.markdown(
                f'<span class="fg-muted">Rule: {comp.get("selection_rule", "")}</span>',
                unsafe_allow_html=True,
            )
        with st.expander("Hyper-parameter search"):
            for name, block in comp["models"].items():
                trials = block.get("training", {}).get("search_trials")
                if trials:
                    st.markdown(
                        f"**{name}** · best: `{block['training'].get('best_params')}` · searched on {block['training'].get('search_rows', 0):,} training rows"
                    )
                    table(
                        pd.json_normalize(trials).rename(columns=lambda c: c.replace("params.", ""))
                    )

    with tab2:
        sweep = reports["threshold_sweep"]
        if sweep is not None:
            plot(threshold_figure(sweep, trained_threshold, threshold))
            row = sweep.iloc[(sweep["threshold"] - threshold).abs().argsort().iloc[0]]
            cards(
                [
                    ("Threshold", f"{row['threshold']:.2f}", "from the sidebar slider"),
                    ("Precision", f"{row['precision']:.4f}", "validation set"),
                    ("Recall", f"{row['recall']:.4f}", "validation set"),
                    ("False positives", f"{int(row['fp']):,}", "validation set"),
                    ("False negatives", f"{int(row['fn']):,}", "validation set"),
                ]
            )
            st.markdown(
                '<span class="fg-muted">Move the sidebar slider to see the trade-off at any threshold. '
                "The dashed line is the trained value.</span>",
                unsafe_allow_html=True,
            )

    with tab3:
        imp = reports["shap_importance"]
        if imp is not None:
            plot(importance_figure(imp))
            st.markdown(
                '<span class="fg-muted">Mean absolute SHAP value over a validation sample. The '
                "balance-reconciliation features dominate: PaySim fraud drains the origin account.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("SHAP importance was not produced for the selected model.")

    with tab4:
        figs = reports["figures_dir"]
        c1, c2 = st.columns(2)
        for col, (name, caption) in zip(
            [c1, c2, c1, c2],
            [
                ("pr_curve_test.png", "Precision–recall curves, test set"),
                ("roc_curve_test.png", "ROC curves, test set"),
                ("confusion_matrix_test.png", "Confusion matrix of the deployed model, test set"),
                ("threshold_analysis.png", "Threshold sweep on validation"),
            ],
            strict=True,
        ):
            f = figs / name
            if f.exists():
                with col:
                    image(f, caption)


# --------------------------------------------------------------------------- #
# Prediction history
# --------------------------------------------------------------------------- #
elif page == "Prediction history":
    st.title("Prediction history")
    st.markdown(
        '<div class="fg-topline">Every score produced by the API or this dashboard is stored in SQLite '
        "with its threshold, decision and reasons.</div>",
        unsafe_allow_html=True,
    )
    try:
        stats = backend.history_stats()
    except Exception as e:
        st.error(f"History unavailable: {e}")
        st.stop()
    cards(
        [
            (
                "Predictions",
                f"{stats.get('total_predictions', 0):,}",
                f"in {stats.get('batches', 0):,} batches",
            ),
            (
                "Flagged",
                f"{stats.get('flagged_fraud', 0):,}",
                pct(stats.get("flag_rate", 0.0)) + " of predictions",
            ),
            (
                "Last scored",
                local_time(pd.Series([stats.get("last_prediction_at")])).iloc[0]
                if stats.get("last_prediction_at")
                else "–",
                "local time",
            ),
        ]
    )
    c1, c2 = st.columns([0.3, 0.7])
    only_fraud = c1.checkbox("Flagged only")
    limit = c2.slider("Rows to show", 10, 1000, 200, 10)
    hist = backend.history(limit=limit, only_fraud=only_fraud)
    if hist.empty:
        st.markdown(
            '<div class="fg-empty">No predictions stored yet.</div>', unsafe_allow_html=True
        )
    else:
        if "reasons" in hist.columns:
            hist["reasons"] = join_reasons(hist["reasons"])
        if "created_at" in hist.columns:
            hist["created_at"] = local_time(hist["created_at"])
        cols = [
            c
            for c in [
                "created_at",
                "source",
                "type",
                "amount",
                "fraud_probability",
                "is_fraud",
                "threshold",
                "risk_level",
                "reasons",
                "batch_id",
            ]
            if c in hist.columns
        ]
        table(hist[cols], height=480)
        st.download_button(
            "Download history CSV",
            hist.to_csv(index=False).encode(),
            "fraudguard_history.csv",
            "text/csv",
        )
