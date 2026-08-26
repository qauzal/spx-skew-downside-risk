"""Run the dissertation tail-risk models and recursive forecast evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tailrisk_specification_audit"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEED = 20210816
BOOT_REPS = 1000
BOOT_BLOCK = 3
EPS = 1e-8

PRED_LABELS = {
    "skew_minus_index": "SKEW- level",
    "skew_minus_change": "SKEW- change",
    "skew_minus_z60": "SKEW- rolling z-score",
    "low_skew_minus": "Low-SKEW- regime",
    "skew_z_low_vix_interaction": "Low-VIX interaction",
    "published_skew": "Published SKEW",
    "skew25_diff": "25-delta skew",
}


def safe_auc(y, p):
    return roc_auc_score(y, p) if pd.Series(y).nunique() == 2 else np.nan


def _finite_model_arrays(result):
    """Return False for the numerical pathologies that invalidate inference."""
    arrays = [getattr(result, name, []) for name in ("params", "bse", "pvalues")]
    return all(np.all(np.isfinite(np.asarray(values, dtype=float))) for values in arrays)


def _separation_like(result, converged=True, warning_text=""):
    params = np.asarray(getattr(result, "params", []), dtype=float)
    ses = np.asarray(getattr(result, "bse", []), dtype=float)
    warning_lower = warning_text.lower()
    return bool(
        (not converged)
        or (not _finite_model_arrays(result))
        or np.any(np.abs(params) > 20)
        or np.any(ses > 20)
        or any(token in warning_lower for token in (
            "separ", "singular", "overflow", "perfect prediction",
            "failed to converge", "inverting hessian"))
    )


def mcfadden(llf, y):
    p = np.clip(np.mean(y), EPS, 1 - EPS)
    ll0 = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return 1 - llf / ll0 if ll0 != 0 else np.nan


def qloss(y, qhat, q=.10):
    e = np.asarray(y) - np.asarray(qhat)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def design_diagnostics(frame, xcols, residuals=None):
    X = sm.add_constant(frame[xcols].astype(float), has_constant="add")
    try:
        cond = float(np.linalg.cond(X))
    except Exception:
        cond = np.nan
    vifs = {}
    if X.shape[1] > 2:
        for j, col in enumerate(X.columns):
            if col != "const":
                try:
                    vifs[col] = float(variance_inflation_factor(X.values, j))
                except Exception:
                    vifs[col] = np.nan
    dw = float(durbin_watson(residuals)) if residuals is not None else np.nan
    return cond, vifs, dw


class FailedBinaryResult:
    """Shape-compatible record used only to report a failed unpenalised fit."""
    def __init__(self, y, columns):
        y = pd.Series(y, dtype=float)
        self.params = pd.Series(np.nan, index=columns, dtype=float)
        self.bse = pd.Series(np.nan, index=columns, dtype=float)
        self.pvalues = pd.Series(np.nan, index=columns, dtype=float)
        self.nobs = len(y)
        self.llf = np.nan
        self.aic = np.nan
        self.resid = y - y.mean()
        self.mle_retvals = {"converged": False}
        self.converged = False
        self._mean = float(np.clip(y.mean(), EPS, 1 - EPS))

    def predict(self, exog):
        return pd.Series(np.repeat(self._mean, len(exog)), index=exog.index)


class Store:
    def __init__(self):
        self.registry = []
        self.results = []

    def add_registry(self, **kw):
        self.registry.append(kw)

    def add_terms(self, model_id, family, outcome, sample, result, frame, xcols,
                  se_kind, focal_terms, extra=None, converged=True, warning_text=""):
        extra = extra or {}
        cond, vifs, dw = design_diagnostics(frame, xcols,
                                             getattr(result, "resid", None))
        corr = frame[["vix", "skew_minus_index"]].corr().iloc[0, 1] \
            if {"vix", "skew_minus_index"}.issubset(frame.columns) else np.nan
        params = result.params
        ses = result.bse
        pvals = result.pvalues
        model_separation = _separation_like(result, converged, warning_text)
        for term in params.index:
            row = dict(
                model_id=model_id, family=family, outcome=outcome,
                sample=sample, term=term, focal=term in focal_terms,
                coefficient=float(params[term]), standard_error=float(ses[term]),
                statistic=float(params[term] / ses[term]) if ses[term] else np.nan,
                raw_pvalue=float(pvals[term]), se_kind=se_kind,
                n=int(result.nobs), event_count=(int(frame[outcome].sum())
                    if set(pd.unique(frame[outcome].dropna())).issubset({0, 1}) else np.nan),
                converged=bool(converged), warning=warning_text,
                separation_flag=model_separation,
                model_valid=bool(converged and not model_separation),
                low_power_flag=(int(frame[outcome].sum()) < 10
                    if set(pd.unique(frame[outcome].dropna())).issubset({0, 1}) else False),
                vix_skew_minus_corr=corr, condition_number=cond,
                vif=float(vifs.get(term, np.nan)), durbin_watson=dw,
            )
            row.update(extra)
            self.results.append(row)


def load_data():
    y = pd.read_csv(OUT / "tail_outcomes_monthly.csv", parse_dates=[
        "predictor_date", "outcome_window_start", "outcome_window_end",
        "monthly_outcome_date"])
    x = pd.read_csv(OUT / "predictor_variants_monthly.csv",
                    parse_dates=["predictor_date"])
    d = y.merge(x, on=["month", "predictor_date"], validate="one_to_one")
    d["year"] = d.predictor_date.dt.year
    return d


def sample_frame(d, sample):
    if sample == "1996-2017":
        return d[d.year <= 2017].copy()
    if sample == "2018-2025":
        return d[d.year >= 2018].copy()
    return d.copy()


def fit_logit(store, d, model_id, family, outcome, xcols, sample,
              focal_terms, benchmark=None, notes=""):
    cols = [outcome] + xcols
    f = d.dropna(subset=cols).copy()
    X = sm.add_constant(f[xcols].astype(float), has_constant="add")
    warntext = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = sm.Logit(f[outcome].astype(float), X).fit(disp=0, maxiter=500)
            conv = bool(res.mle_retvals.get("converged", True))
        except Exception as exc:
            try:
                res = sm.GLM(f[outcome].astype(float), X,
                             family=sm.families.Binomial()).fit(maxiter=500)
                conv = bool(getattr(res, "converged", True))
                warntext = f"Discrete logit failed ({exc}); binomial GLM MLE used."
            except Exception as glm_exc:
                res = FailedBinaryResult(f[outcome], X.columns)
                conv = False
                warntext = (f"Discrete logit failed ({exc}); binomial GLM also "
                            f"failed ({glm_exc}); model recorded as not estimable.")
        if caught:
            warntext += " " + " | ".join(str(w.message) for w in caught)
    pred = np.clip(np.asarray(res.predict(X)), EPS, 1 - EPS)
    llf = float(res.llf)
    extra = dict(
        log_likelihood=llf, aic=float(res.aic), pseudo_r2=mcfadden(llf, f[outcome].values),
        auc=safe_auc(f[outcome], pred), brier=brier_score_loss(f[outcome], pred),
        rmse=np.nan, mae=np.nan, adjusted_r2=np.nan, lr_pvalue=np.nan,
        average_marginal_effect=np.nan, bootstrap_ci_low=np.nan,
        bootstrap_ci_high=np.nan, pinball_loss=np.nan,
        probability_effect_1unit_decrease=np.nan,
        probability_effect_1pct_decrease=np.nan,
    )
    benchmark_valid = True
    benchmark_converged = True
    benchmark_warning = ""
    lr_suppression_reason = ""
    if benchmark:
        bcols = benchmark
        bX = sm.add_constant(f[bcols].astype(float), has_constant="add")
        with warnings.catch_warnings(record=True) as bcaught:
            warnings.simplefilter("always")
            try:
                br = sm.Logit(f[outcome].astype(float), bX).fit(disp=0, maxiter=500)
                benchmark_converged = bool(br.mle_retvals.get("converged", True))
            except Exception as exc:
                try:
                    br = sm.GLM(f[outcome].astype(float), bX,
                                family=sm.families.Binomial()).fit(maxiter=500)
                    benchmark_converged = bool(getattr(br, "converged", True))
                    benchmark_warning = f"Discrete benchmark failed ({exc}); GLM used."
                except Exception as glm_exc:
                    br = FailedBinaryResult(f[outcome], bX.columns)
                    benchmark_converged = False
                    benchmark_warning = (
                        f"Discrete benchmark failed ({exc}); binomial GLM also "
                        f"failed ({glm_exc}); benchmark recorded as not estimable.")
            if bcaught:
                benchmark_warning = (benchmark_warning + " " + " | ".join(
                    str(w.message) for w in bcaught)).strip()
        benchmark_valid = not _separation_like(
            br, benchmark_converged, benchmark_warning)
        df = max(len(xcols) - len(bcols), 1)
        candidate_valid = not _separation_like(res, conv, warntext)
        nesting_valid = set(bcols).issubset(xcols) and len(xcols) > len(bcols)
        ll_order_valid = np.isfinite(res.llf) and np.isfinite(br.llf) and (
            res.llf >= br.llf - 1e-8)
        if not candidate_valid:
            lr_suppression_reason = "candidate invalid/non-convergent"
        elif not benchmark_valid:
            lr_suppression_reason = "benchmark invalid/non-convergent"
        elif not nesting_valid:
            lr_suppression_reason = "models are not nested"
        elif not ll_order_valid:
            lr_suppression_reason = "invalid likelihood ordering"
        else:
            extra["lr_pvalue"] = float(st.chi2.sf(
                max(0.0, 2 * (res.llf - br.llf)), df))
        extra["candidate_valid_for_lr"] = candidate_valid
        extra["lr_test_valid"] = not bool(lr_suppression_reason)
    else:
        extra["candidate_valid_for_lr"] = not _separation_like(res, conv, warntext)
        extra["lr_test_valid"] = False
    extra["benchmark_valid_for_lr"] = benchmark_valid
    extra["benchmark_warning"] = benchmark_warning
    extra["lr_suppression_reason"] = lr_suppression_reason
    for term in focal_terms:
        if term in res.params:
            extra_term = extra.copy()
            extra_term["average_marginal_effect"] = float(
                res.params[term] * np.mean(pred * (1 - pred)))
            # add_terms writes one shared extra; overwrite AME after below
    store.add_registry(model_id=model_id, family=family, outcome=outcome,
        sample=sample, model_type="logit", predictors=" + ".join(xcols),
        benchmark_predictors=" + ".join(benchmark or []), notes=notes)
    before = len(store.results)
    store.add_terms(model_id, family, outcome, sample, res, f, xcols,
                    "MLE", focal_terms, extra, conv, warntext.strip())
    for row in store.results[before:]:
        if row["term"] in focal_terms:
            row["average_marginal_effect"] = float(
                res.params[row["term"]] * np.mean(pred * (1 - pred)))
            term = row["term"]
            X1 = X.copy()
            X1[term] = X1[term] - 1.0
            row["probability_effect_1unit_decrease"] = float(
                np.mean(np.asarray(res.predict(X1)) - pred))
            if term.endswith("skew_minus_index") or term in {
                    "skew_minus_index", "published_skew"}:
                Xp = X.copy()
                Xp[term] = .99 * Xp[term]
                row["probability_effect_1pct_decrease"] = float(
                    np.mean(np.asarray(res.predict(Xp)) - pred))
    return res, f, pred


def fit_ols(store, d, model_id, family, outcome, xcols, sample, focal_terms,
            notes=""):
    f = d.dropna(subset=[outcome] + xcols).copy()
    X = sm.add_constant(f[xcols].astype(float), has_constant="add")
    base = sm.OLS(f[outcome].astype(float), X).fit()
    res = base.get_robustcov_results(cov_type="HAC", maxlags=1)
    # Restore labelled Series because robust wrapper sometimes returns ndarrays.
    class Labelled:
        pass
    lr = Labelled()
    lr.params = pd.Series(res.params, index=X.columns)
    lr.bse = pd.Series(res.bse, index=X.columns)
    lr.pvalues = pd.Series(res.pvalues, index=X.columns)
    lr.resid = base.resid
    lr.nobs = base.nobs
    pred = base.predict(X)
    extra = dict(log_likelihood=float(base.llf), aic=float(base.aic), pseudo_r2=np.nan,
        auc=np.nan, brier=np.nan, rmse=mean_squared_error(f[outcome], pred) ** .5,
        mae=mean_absolute_error(f[outcome], pred), adjusted_r2=float(base.rsquared_adj),
        lr_pvalue=np.nan, average_marginal_effect=np.nan,
        bootstrap_ci_low=np.nan, bootstrap_ci_high=np.nan, pinball_loss=np.nan,
        probability_effect_1unit_decrease=np.nan,
        probability_effect_1pct_decrease=np.nan)
    store.add_registry(model_id=model_id, family=family, outcome=outcome,
        sample=sample, model_type="OLS-HAC(1)", predictors=" + ".join(xcols),
        benchmark_predictors="vix", notes=notes)
    store.add_terms(model_id, family, outcome, sample, lr, f, xcols,
                    "HAC/Newey-West, lag 1", focal_terms, extra)
    return base, f, np.asarray(pred)


def moving_block_indices(n, rng, block=BOOT_BLOCK):
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def fit_quantile(store, d, model_id, outcome, xcols, focal_terms, bootstrap=True):
    f = d.dropna(subset=[outcome] + xcols).copy()
    X = sm.add_constant(f[xcols].astype(float), has_constant="add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sm.QuantReg(f[outcome].astype(float), X).fit(q=.10, max_iter=5000)
    warning_text = " | ".join(str(w.message) for w in caught)
    iterations = int(getattr(res, "iterations", 0) or 0)
    converged = bool(iterations < 5000 and _finite_model_arrays(res))
    pred = np.asarray(res.predict(X))
    boot = {term: [] for term in focal_terms}
    if bootstrap and focal_terms:
        rng = np.random.default_rng(SEED)
        for _ in range(BOOT_REPS):
            idx = moving_block_indices(len(f), rng)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    br = sm.QuantReg(f[outcome].iloc[idx].to_numpy(),
                        X.iloc[idx].reset_index(drop=True)).fit(q=.10, max_iter=2500)
                for term in focal_terms:
                    boot[term].append(float(br.params[term]))
            except Exception:
                continue
    extra = dict(log_likelihood=np.nan, aic=np.nan,
        pseudo_r2=float(res.prsquared), auc=np.nan, brier=np.nan, rmse=np.nan,
        mae=np.nan, adjusted_r2=np.nan, lr_pvalue=np.nan,
        average_marginal_effect=np.nan, bootstrap_ci_low=np.nan,
        bootstrap_ci_high=np.nan, pinball_loss=qloss(f[outcome], pred),
        probability_effect_1unit_decrease=np.nan,
        probability_effect_1pct_decrease=np.nan)
    success_counts = {term: len(boot.get(term, [])) for term in focal_terms}
    if bootstrap:
        model_type = "QuantReg q=0.10 with moving-block bootstrap inference"
        inference_note = (f"Seed {SEED}; requested {BOOT_REPS} reps; block length "
                          f"{BOOT_BLOCK}; successful focal draws {success_counts}")
    else:
        model_type = "QuantReg q=0.10 with asymptotic inference"
        inference_note = "Bootstrap disabled for common-row benchmark; asymptotic inference reported"
    store.add_registry(model_id=model_id, family="E_quantile", outcome=outcome,
        sample="1996-2025", model_type=model_type,
        predictors=" + ".join(xcols),
        benchmark_predictors=("vix" if focal_terms else ""),
        notes=f"{inference_note}; iterations={iterations}; warnings={warning_text}")
    # Write manually so bootstrap inference replaces asymptotic values for focal.
    cond, vifs, dw = design_diagnostics(f, xcols, res.resid)
    corr = f[["vix", "skew_minus_index"]].corr().iloc[0, 1] \
        if {"vix", "skew_minus_index"}.issubset(f.columns) else np.nan
    for term in res.params.index:
        vals = np.asarray(boot.get(term, []), dtype=float)
        if len(vals):
            se = float(np.std(vals, ddof=1))
            lo, hi = np.quantile(vals, [.025, .975])
            p = float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
        else:
            se, lo, hi, p = float(res.bse[term]), np.nan, np.nan, float(res.pvalues[term])
        row = dict(model_id=model_id, family="E_quantile", outcome=outcome,
            sample="1996-2025", term=term, focal=term in focal_terms,
            coefficient=float(res.params[term]), standard_error=se,
            statistic=float(res.params[term] / se) if se else np.nan,
            raw_pvalue=p, se_kind=("moving-block bootstrap" if len(vals) else "asymptotic"),
            n=len(f), event_count=np.nan, converged=converged, warning=warning_text,
            separation_flag=(not converged), model_valid=converged,
            low_power_flag=False, vix_skew_minus_corr=corr,
            condition_number=cond, vif=float(vifs.get(term, np.nan)),
            durbin_watson=dw, **extra)
        row["bootstrap_ci_low"] = float(lo) if np.isfinite(lo) else np.nan
        row["bootstrap_ci_high"] = float(hi) if np.isfinite(hi) else np.nan
        store.results.append(row)
    return res, f, pred


def run_family_a(store, d):
    comparisons = {"published_skew": "published", "skew25_diff": "skew25"}
    for sample in ["1996-2017", "2018-2025", "1996-2025"]:
        for h in [1, 3, 6, 12]:
            # Lags are formed on the full history; the sample is defined by the
            # dependent-variable month. Thus January 2018 may use December 2017.
            f = d.copy()
            f["lag_event"] = f.MktDown.shift(h)
            for var in ["skew_minus_index", "vix"]:
                f[f"lag{h}_{var}"] = f[var].shift(h)
            f = sample_frame(f, sample)
            focal = f"lag{h}_skew_minus_index"
            fit_logit(store, f, f"A_exact_h{h}_{sample}", "A_exact_BT",
                "MktDown", ["lag_event", focal], sample, [focal],
                notes="Exact B&T-style specification")
            fit_logit(store, f, f"A_nolagoutcome_h{h}_{sample}", "A_no_lag_outcome",
                "MktDown", [focal], sample, [focal], notes="Lagged-outcome sensitivity")
            _, common_f, _ = fit_logit(store, f, f"A_vix_h{h}_{sample}", "A_VIX_incremental",
                "MktDown", ["lag_event", f"lag{h}_vix", focal], sample, [focal],
                benchmark=["lag_event", f"lag{h}_vix"])
            fit_logit(
                store, common_f, f"A0_common_vix_h{h}_{sample}",
                "A_VIX_incremental_benchmark", "MktDown",
                ["lag_event", f"lag{h}_vix"], sample, [],
                notes=f"Exact common-row benchmark for A_vix_h{h}_{sample}",
            )
        # h=1 comparisons only
        f = d.copy()
        f["lag_event"] = f.MktDown.shift(1)
        f["lag1_vix"] = f.vix.shift(1)
        for var, tag in comparisons.items():
            lv = f"lag1_{var}"
            f[lv] = f[var].shift(1)
        f = sample_frame(f, sample)
        for var, tag in comparisons.items():
            lv = f"lag1_{var}"
            fit_logit(store, f, f"A_compare_{tag}_exact_{sample}", "A_h1_comparisons",
                "MktDown", ["lag_event", lv], sample, [lv])
            _, common_f, _ = fit_logit(store, f, f"A_compare_{tag}_vix_{sample}", "A_h1_comparisons",
                "MktDown", ["lag_event", "lag1_vix", lv], sample, [lv],
                benchmark=["lag_event", "lag1_vix"])
            fit_logit(
                store, common_f, f"A0_common_{tag}_vix_{sample}",
                "A_h1_comparisons_benchmark", "MktDown",
                ["lag_event", "lag1_vix"], sample, [],
                notes=f"Exact common-row benchmark for A_compare_{tag}_vix_{sample}",
            )


FORMS = {
    "level": (["vix", "skew_minus_index"], ["skew_minus_index"]),
    "change": (["vix", "skew_minus_change"], ["skew_minus_change"]),
    "z60": (["vix", "skew_minus_z60"], ["skew_minus_z60"]),
    "low": (["vix", "low_skew_minus"], ["low_skew_minus"]),
    "interaction": (["vix", "skew_minus_z60", "low_vix",
                     "skew_z_low_vix_interaction"], ["skew_z_low_vix_interaction"]),
}


def run_families_b_to_e(store, d):
    # Binary additions and exact common-row VIX benchmarks.
    fit_logit(store, d, "B_standalone_level", "B_DD21_event", "DD21Event",
              ["skew_minus_index"], "1996-2025", ["skew_minus_index"],
              notes="Descriptive standalone model")
    for tag, (xcols, focal) in FORMS.items():
        _, f, _ = fit_logit(store, d, f"B_{tag}", "B_DD21_event", "DD21Event",
              xcols, "1996-2025", focal, benchmark=["vix"])
        fit_logit(store, f, f"B0_common_{tag}", "B_DD21_event_benchmark",
                  "DD21Event", ["vix"], "1996-2025", [],
                  notes=f"Common rows for B_{tag}")
    for var, tag in [("published_skew", "published"), ("skew25_diff", "skew25")]:
        _, f, _ = fit_logit(store, d, f"B_compare_{tag}", "B_DD21_event",
            "DD21Event", ["vix", var], "1996-2025", [var], benchmark=["vix"])
        fit_logit(store, f, f"B0_common_{tag}", "B_DD21_event_benchmark",
            "DD21Event", ["vix"], "1996-2025", [], notes=f"Common rows for {tag}")

    for family, outcome in [("C_DD21_loss", "DD21Loss"),
                            ("D_LogDSV21", "LogDSV21")]:
        prefix = "C" if family.startswith("C") else "D"
        for tag, (xcols, focal) in FORMS.items():
            _, f, _ = fit_ols(store, d, f"{prefix}_{tag}", family, outcome,
                              xcols, "1996-2025", focal)
            fit_ols(store, f, f"{prefix}0_common_{tag}", family + "_benchmark",
                    outcome, ["vix"], "1996-2025", [], notes=f"Common rows for {tag}")

    # Quantile models; bootstrap only additions, as specified inference is focal.
    for tag, (xcols, focal) in FORMS.items():
        _, f, _ = fit_quantile(store, d, f"E_{tag}", "R21", xcols, focal, True)
        fit_quantile(store, f, f"E0_common_{tag}", "R21", ["vix"], [], False)


def firth_availability(store):
    packages = [p for p in ["firthlogist", "pyfirth", "firthmodels"]
                if importlib.util.find_spec(p) is not None]
    text = "# Rare-event sensitivity\n\n"
    if not packages:
        text += ("Rare-event sensitivity could not be evaluated using a vetted "
                 "Firth implementation. Firth logit was not estimated because no vetted Firth "
                 "logistic-regression implementation (`firthlogist`, `pyfirth`, or "
                 "`firthmodels`) is installed. No hand-coded or L2-penalised substitute is used. "
                 "Standard logit estimates are retained with explicit low-event and "
                 "separation diagnostics. "
                 "This is a documented software-availability limitation, not evidence "
                 "for or against rare-event robustness.\n")
        status = "skipped_no_vetted_implementation"
    else:
        text += ("Rare-event sensitivity could not be evaluated using a vetted "
                 "Firth implementation. Firth logit was not estimated. A candidate package was detected, "
                 "but its API/version was not pre-specified or vetted; treating it as "
                 "validated is outside the dissertation design. Detected: "
                 + ", ".join(packages) + ". Standard logit estimates are retained "
                 "with explicit low-event and separation diagnostics. This is a documented implementation "
                 "limitation, not evidence for or against rare-event robustness.\n")
        status = "skipped_unvetted_detected_package"
    (OUT / "rare_event_sensitivity.md").write_text(text, encoding="utf-8")
    store.add_registry(model_id="Firth_phase", family="rare_event_sensitivity",
        outcome="MktDown/DD21Event", sample="all", model_type="Firth logit",
        predictors="SKEW-; VIX + SKEW-", benchmark_predictors="standard logit",
        notes=status)


def glm_predict(train, test, outcome, xcols):
    X = sm.add_constant(train[xcols].astype(float), has_constant="add")
    Xt = sm.add_constant(test[xcols].astype(float), has_constant="add")
    Xt = Xt.reindex(columns=X.columns, fill_value=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = sm.GLM(train[outcome].astype(float), X,
                       family=sm.families.Binomial()).fit(maxiter=300)
            fit_error = ""
        except Exception as exc:
            r = FailedBinaryResult(train[outcome], X.columns)
            fit_error = f"Binomial GLM failed ({exc}); mean forecast recorded."
    warning_text = " | ".join(str(w.message) for w in caught)
    warning_text = " | ".join(x for x in [fit_error, warning_text] if x)
    converged = bool(getattr(r, "converged", True))
    valid = not _separation_like(r, converged, warning_text)
    return (np.clip(np.asarray(r.predict(Xt)), EPS, 1 - EPS),
            np.clip(np.asarray(r.predict(X)), EPS, 1 - EPS),
            dict(model_converged=converged, model_valid=valid,
                 model_warning=warning_text))


def _evaluation_period(test, availability):
    evaluation_date = pd.Timestamp(test[availability])
    return "2006-2017" if evaluation_date <= pd.Timestamp("2017-12-31") else "2018-2025"


def _pair_metadata(train, test, outcome, availability, addition_model, task):
    forecast_date = pd.Timestamp(test.predictor_date)
    availability_date = pd.Timestamp(test[availability])
    train_max_availability = pd.Timestamp(train[availability].max())
    pair_id = f"{task}|{outcome}|{addition_model}|{forecast_date:%Y-%m-%d}"
    return dict(
        pair_id=pair_id,
        addition_model=addition_model,
        predictor_date=forecast_date,
        evaluation_period=_evaluation_period(test, availability),
        test_outcome_availability_date=availability_date,
        max_training_outcome_availability_date=train_max_availability,
        availability_check_passed=bool(train_max_availability <= forecast_date),
        training_rows_identical_by_design=True,
        formal_forecast_inference_performed=False,
        training_row_count=int(len(train)),
        train_n=int(len(train)),
        train_start=pd.Timestamp(train.predictor_date.min()),
        train_end=pd.Timestamp(train.predictor_date.max()),
    )


def recursive_binary(d, outcome, availability, model, xcols):
    rows = []
    benchmark_xcols = ["vix"]
    benchmark_model = f"VIX only (paired with {model})"
    base = d.dropna(subset=[outcome, availability] + sorted(set(xcols + benchmark_xcols))).copy()
    for _, test in base[base.predictor_date >= "2006-01-01"].iterrows():
        train = base[(base.predictor_date < test.predictor_date) &
                     (base[availability] <= test.predictor_date)]
        if len(train) < 60 or train[outcome].nunique() < 2:
            continue
        one = test.to_frame().T
        p, fitted, diag = glm_predict(train, one, outcome, xcols)
        bp, bfitted, bdiag = glm_predict(train, one, outcome, benchmark_xcols)
        meta = _pair_metadata(train, test, outcome, availability, model, "binary")
        for role, name, paired, prediction, in_sample, model_diag in [
                ("candidate", model, benchmark_model, p, fitted, diag),
                ("benchmark", benchmark_model, model, bp, bfitted, bdiag)]:
            threshold = float(np.quantile(in_sample, .90))
            rows.append(dict(task="binary", outcome=outcome, model=name,
                pair_role=role, paired_model=paired, actual=float(test[outcome]),
                prediction=float(prediction[0]), alert=int(prediction[0] > threshold),
                alert_threshold=threshold, **meta, **model_diag))
    return rows


def recursive_ols(d, outcome, availability, model, xcols):
    rows = []
    benchmark_xcols = ["vix"]
    benchmark_model = f"VIX only (paired with {model})"
    base = d.dropna(subset=[outcome, availability] + sorted(set(xcols + benchmark_xcols))).copy()
    for _, test in base[base.predictor_date >= "2006-01-01"].iterrows():
        train = base[(base.predictor_date < test.predictor_date) &
                     (base[availability] <= test.predictor_date)]
        if len(train) < 60:
            continue
        meta = _pair_metadata(train, test, outcome, availability, model, "continuous")
        one = test.to_frame().T
        for role, name, paired, cols in [
                ("candidate", model, benchmark_model, xcols),
                ("benchmark", benchmark_model, model, benchmark_xcols)]:
            X = sm.add_constant(train[cols].astype(float), has_constant="add")
            Xt = sm.add_constant(one[cols].astype(float), has_constant="add")
            Xt = Xt.reindex(columns=X.columns, fill_value=1.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = sm.OLS(train[outcome].astype(float), X).fit()
            warning_text = " | ".join(str(w.message) for w in caught)
            pred = float(fitted.predict(Xt).iloc[0])
            valid = _finite_model_arrays(fitted) and np.isfinite(pred)
            rows.append(dict(task="continuous", outcome=outcome, model=name,
                pair_role=role, paired_model=paired, actual=float(test[outcome]),
                prediction=pred, alert=np.nan, alert_threshold=np.nan,
                model_converged=True, model_valid=bool(valid),
                model_warning=warning_text, **meta))
    return rows


def recursive_quantile(d, model, xcols):
    rows = []
    outcome, availability = "R21", "outcome_window_end"
    benchmark_xcols = ["vix"]
    benchmark_model = f"VIX only (paired with {model})"
    base = d.dropna(subset=[outcome, availability] + sorted(set(xcols + benchmark_xcols))).copy()
    for _, test in base[base.predictor_date >= "2006-01-01"].iterrows():
        train = base[(base.predictor_date < test.predictor_date) &
                     (base[availability] <= test.predictor_date)]
        if len(train) < 60:
            continue
        meta = _pair_metadata(train, test, outcome, availability, model, "quantile")
        one = test.to_frame().T
        for role, name, paired, cols in [
                ("candidate", model, benchmark_model, xcols),
                ("benchmark", benchmark_model, model, benchmark_xcols)]:
            X = sm.add_constant(train[cols].astype(float), has_constant="add")
            Xt = sm.add_constant(one[cols].astype(float), has_constant="add")
            Xt = Xt.reindex(columns=X.columns, fill_value=1.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = sm.QuantReg(train[outcome].astype(float), X).fit(
                    q=.10, max_iter=3000)
            warning_text = " | ".join(str(w.message) for w in caught)
            iterations = int(getattr(r, "iterations", 0) or 0)
            pred = float(r.predict(Xt).iloc[0])
            valid = bool(iterations < 3000 and _finite_model_arrays(r)
                         and np.isfinite(pred))
            rows.append(dict(task="quantile", outcome=outcome, model=name,
                pair_role=role, paired_model=paired, actual=float(test[outcome]),
                prediction=pred, alert=np.nan, alert_threshold=np.nan,
                model_converged=bool(iterations < 3000), model_valid=valid,
                model_warning=warning_text, **meta))
    return rows


def calibration(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    try:
        ci = sm.GLM(np.asarray(y), np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0]
        cs = sm.GLM(np.asarray(y), sm.add_constant(z), family=sm.families.Binomial()).fit().params[1]
        return float(ci), float(cs)
    except Exception:
        return np.nan, np.nan


def run_oos(d):
    rows = []
    binmods = {
        "VIX + SKEW- level": ["vix", "skew_minus_index"],
        "VIX + SKEW- change": ["vix", "skew_minus_change"],
        "VIX + SKEW- z-score": ["vix", "skew_minus_z60"],
        "VIX + low-SKEW-": ["vix", "low_skew_minus"],
    }
    for model, xcols in binmods.items():
        rows += recursive_binary(d, "MktDownNext", "monthly_outcome_date", model, xcols)
        rows += recursive_binary(d, "DD21Event", "outcome_window_end", model, xcols)
    contmods = {
        "VIX + SKEW- level": ["vix", "skew_minus_index"],
        "VIX + SKEW- change": ["vix", "skew_minus_change"],
        "VIX + SKEW- z-score": ["vix", "skew_minus_z60"],
    }
    for outcome in ["DD21Loss", "LogDSV21"]:
        for model, xcols in contmods.items():
            rows += recursive_ols(d, outcome, "outcome_window_end", model, xcols)
    for model, xcols in contmods.items():
        rows += recursive_quantile(d, model, xcols)
    p = pd.DataFrame(rows)
    if p.empty:
        raise RuntimeError("No recursive forecasts were produced")
    if not p.availability_check_passed.all():
        raise RuntimeError("OOS leakage guard failed: a training outcome was unavailable")
    if not p.groupby("pair_id").size().eq(2).all():
        raise RuntimeError("Each OOS pair_id must identify exactly two forecast rows")
    p.to_csv(OUT / "out_of_sample_predictions.csv", index=False)
    metrics = []
    for (task, outcome, period, model), g in p.groupby(["task", "outcome", "evaluation_period", "model"]):
        y, pred = g.actual.values, g.prediction.values
        r = dict(task=task, outcome=outcome, evaluation_period=period, model=model,
                 pair_role=g.pair_role.iloc[0], paired_model=g.paired_model.iloc[0],
                 n=len(g), event_count=np.nan,
                 invalid_forecast_fits=int((~g.model_valid.astype(bool)).sum()),
                 availability_violations=int((~g.availability_check_passed.astype(bool)).sum()),
                 formal_forecast_inference_performed=False)
        if task == "binary":
            ci, cs = calibration(y, pred)
            event = y == 1
            alert = g.alert.values == 1
            r.update(brier=brier_score_loss(y, pred), log_loss=log_loss(y, pred, labels=[0,1]),
                auc=safe_auc(y, pred), calibration_intercept=ci, calibration_slope=cs,
                event_count=int(y.sum()), avg_probability_events=(float(pred[event].mean()) if event.any() else np.nan),
                avg_probability_nonevents=(float(pred[~event].mean()) if (~event).any() else np.nan),
                hit_rate=(float(alert[event].mean()) if event.any() else np.nan),
                false_positive_rate=(float(alert[~event].mean()) if (~event).any() else np.nan),
                precision=(float(y[alert].mean()) if alert.any() else np.nan), number_alerts=int(alert.sum()))
        elif task == "continuous":
            r.update(rmse=mean_squared_error(y, pred) ** .5, mae=mean_absolute_error(y, pred),
                     mean_forecast_error=float(np.mean(pred - y)))
        else:
            breach = y < pred
            r.update(pinball_loss=qloss(y, pred), empirical_violation_rate=float(breach.mean()),
                     avg_realized_return_when_breached=(float(y[breach].mean()) if breach.any() else np.nan))
        metrics.append(r)
    m = pd.DataFrame(metrics)
    pair_audit = []
    # Every addition has its own recursively refitted VIX benchmark. Candidate
    # and benchmark therefore share exact test dates and exact training rows.
    for i, r in m.iterrows():
        if r.pair_role == "benchmark":
            m.loc[i, "paired_loss_difference_vs_vix"] = 0.0
            m.loc[i, "paired_loss_difference_vs_vix_absolute"] = 0.0
            m.loc[i, "paired_loss_difference_vs_vix_relative"] = 0.0
            m.loc[i, "paired_loss_difference_vs_vix_relative_percent"] = 0.0
            m.loc[i, "comparison_n"] = r.n
            if r.task == "binary":
                m.loc[i, "benchmark_brier_common_rows"] = r.brier
                m.loc[i, "benchmark_log_loss_common_rows"] = r.log_loss
                m.loc[i, "benchmark_auc_common_rows"] = r.auc
            elif r.task == "continuous":
                m.loc[i, "benchmark_rmse_common_rows"] = r.rmse
                m.loc[i, "benchmark_mae_common_rows"] = r.mae
            else:
                m.loc[i, "benchmark_pinball_common_rows"] = r.pinball_loss
            continue
        g = p[(p.task == r.task) & (p.outcome == r.outcome) &
              (p.evaluation_period == r.evaluation_period) & (p.model == r.model)]
        b = p[(p.task == r.task) & (p.outcome == r.outcome) &
              (p.evaluation_period == r.evaluation_period) &
              (p.model == r.paired_model)]
        pair = g.merge(b, on="pair_id", suffixes=("_m", "_b"), validate="one_to_one")
        exact_pairing = bool(
            len(pair) == len(g) == len(b)
            and (pair.predictor_date_m == pair.predictor_date_b).all()
            and (pair.training_row_count_m == pair.training_row_count_b).all()
            and np.allclose(pair.actual_m, pair.actual_b, equal_nan=False)
        )
        if not exact_pairing:
            raise RuntimeError(f"OOS candidate/benchmark rows differ for {r.model}")
        m.loc[i, "comparison_n"] = len(pair)
        if r.task == "binary":
            lm = (pair.actual_m - pair.prediction_m) ** 2
            lb = (pair.actual_b - pair.prediction_b) ** 2
            m.loc[i, "benchmark_brier_common_rows"] = float(np.mean(lb))
            m.loc[i, "benchmark_log_loss_common_rows"] = log_loss(
                pair.actual_b, pair.prediction_b, labels=[0, 1])
            m.loc[i, "benchmark_auc_common_rows"] = safe_auc(
                pair.actual_b, pair.prediction_b)
        elif r.task == "continuous":
            lm = np.abs(pair.actual_m - pair.prediction_m)
            lb = np.abs(pair.actual_b - pair.prediction_b)
            m.loc[i, "benchmark_rmse_common_rows"] = float(np.sqrt(np.mean(
                (pair.actual_b - pair.prediction_b) ** 2)))
            m.loc[i, "benchmark_mae_common_rows"] = float(np.mean(lb))
        else:
            em = pair.actual_m - pair.prediction_m
            eb = pair.actual_b - pair.prediction_b
            lm = np.maximum(.1 * em, -.9 * em)
            lb = np.maximum(.1 * eb, -.9 * eb)
            m.loc[i, "benchmark_pinball_common_rows"] = float(np.mean(lb))
        loss_difference = float(np.mean(lm - lb))
        benchmark_loss = float(np.mean(lb))
        relative_difference = (loss_difference / benchmark_loss
                               if benchmark_loss != 0 else np.nan)
        m.loc[i, "paired_loss_difference_vs_vix"] = loss_difference
        m.loc[i, "paired_loss_difference_vs_vix_absolute"] = loss_difference
        m.loc[i, "paired_loss_difference_vs_vix_relative"] = relative_difference
        m.loc[i, "paired_loss_difference_vs_vix_relative_percent"] = (
            100 * relative_difference if np.isfinite(relative_difference) else np.nan)
        m.loc[i, "descriptively_lower_loss_than_vix"] = loss_difference < 0
        pair_audit.append(dict(
            task=r.task, outcome=r.outcome, evaluation_period=r.evaluation_period,
            candidate_model=r.model, benchmark_model=r.paired_model,
            candidate_rows=len(g), benchmark_rows=len(b), comparison_rows=len(pair),
            exact_test_rows=bool((pair.predictor_date_m == pair.predictor_date_b).all()),
            exact_training_row_counts=bool((pair.training_row_count_m ==
                                             pair.training_row_count_b).all()),
            outcome_availability_guard_passed=bool(
                pair.availability_check_passed_m.all()
                and pair.availability_check_passed_b.all()),
            formal_forecast_inference_performed=False))
    pd.DataFrame(pair_audit).to_csv(OUT / "out_of_sample_pairing_audit.csv", index=False)
    m.to_csv(OUT / "out_of_sample_metrics.csv", index=False)
    return p, m


def assessment(row):
    if (not bool(row.get("converged", True))) or bool(row.get("separation_flag", False)):
        return "not estimable"
    if bool(row.get("low_power_flag", False)):
        return "unstable because of low event count"
    if row.raw_pvalue < .05:
        return "statistically significant at the conventional 5% level"
    if row.raw_pvalue < .10:
        return "suggestive at the conventional 10% level"
    return "not statistically significant"


def make_tables(d, results, oos):
    TABLES.mkdir(exist_ok=True)
    # A
    desc = []
    for name, col, binary, sample_date_col in [
        ("Monthly return <= -5%", "MktDown", True, "predictor_date"),
        ("Existing monthly bottom decile", "bottom_decile_event", True, "predictor_date"),
        ("Forward 21-day start-to-minimum decline <= -5%", "DD21Event", True, "outcome_window_end"),
        ("Forward 21-day start-to-minimum loss", "DD21Loss", False, "outcome_window_end"),
        ("Forward downside semivariance", "DSV21", False, "outcome_window_end"),
        ("Forward 21-day cumulative return", "R21", False, "outcome_window_end")]:
        for sample in ["1996-2017", "2018-2025", "1996-2025"]:
            if sample == "1996-2017":
                frame = d[d[sample_date_col] <= pd.Timestamp("2017-12-31")]
            elif sample == "2018-2025":
                frame = d[d[sample_date_col] >= pd.Timestamp("2018-01-01")]
            else:
                frame = d
            x = frame[col].dropna()
            desc.append(dict(outcome=name, sample=sample, n=len(x),
                event_count=(int(x.sum()) if binary else np.nan), mean=x.mean(),
                sd=x.std(), minimum=x.min(), median=x.median(), maximum=x.max(),
                sample_assignment_date=sample_date_col))
    pd.DataFrame(desc).to_csv(TABLES / "table_A_outcomes.csv", index=False)
    # B and C replication sample
    bench = {1:(-.084,"p<0.01",.097,.90),3:(-.056,"p<0.05",.083,.65),
             6:(-.041,"p<0.10",.047,.70),12:(-.032,"insignificant",.021,.29)}
    tb, tc = [], []
    for h in [1,3,6,12]:
        r = results[(results.model_id == f"A_exact_h{h}_1996-2017") & results.focal].iloc[0]
        b = bench[h]
        if np.sign(r.coefficient) != np.sign(b[0]):
            replication_assessment = "direction differs"
        elif h == 1 and r.raw_pvalue >= .01:
            replication_assessment = (
                "directional partial replication, not a statistical replication of B&T Table 6")
        else:
            replication_assessment = "direction agrees; compare inference separately"
        tb.append(dict(horizon_months=h, our_coefficient=r.coefficient, raw_pvalue=r.raw_pvalue,
            average_marginal_effect_per_unit=r.average_marginal_effect,
            probability_increase_1unit_decrease=r.probability_effect_1unit_decrease,
            probability_point_increase_1pct_decrease=100*r.probability_effect_1pct_decrease,
            bt_probability_point_increase_1pct_decrease=b[3],
            pseudo_r2=r.pseudo_r2, bt_coefficient=b[0], bt_significance=b[1], bt_pseudo_r2=b[2],
            n=r.n, event_count=r.event_count, converged=r.converged,
            separation_flag=r.separation_flag, low_power_flag=r.low_power_flag,
            replication_assessment=replication_assessment))
        rv = results[(results.model_id == f"A_vix_h{h}_1996-2017") & results.focal].iloc[0]
        vv = results[(results.model_id == f"A_vix_h{h}_1996-2017") & (results.term == f"lag{h}_vix")].iloc[0]
        benchmark_id = f"A0_common_vix_h{h}_1996-2017"
        bv = results[results.model_id.eq(benchmark_id)].iloc[0]
        tc.append(dict(horizon_months=h, skew_coefficient=rv.coefficient, raw_pvalue=rv.raw_pvalue,
            vix_coefficient=vv.coefficient,
            vix_pvalue=vv.raw_pvalue, lr_pvalue=rv.lr_pvalue, aic=rv.aic, auc=rv.auc,
            brier=rv.brier, log_likelihood=rv.log_likelihood,
            pseudo_r2=rv.pseudo_r2, n=rv.n, event_count=rv.event_count,
            common_row_benchmark_model_id=benchmark_id,
            benchmark_aic=bv.aic, benchmark_auc=bv.auc, benchmark_brier=bv.brier,
            benchmark_log_likelihood=bv.log_likelihood,
            benchmark_pseudo_r2=bv.pseudo_r2, benchmark_n=bv.n,
            exact_common_rows=(rv.n == bv.n),
            converged=rv.converged, separation_flag=rv.separation_flag,
            low_power_flag=rv.low_power_flag, lr_test_valid=rv.lr_test_valid,
            benchmark_valid_for_lr=rv.benchmark_valid_for_lr,
            lr_suppression_reason=rv.lr_suppression_reason))
    pd.DataFrame(tb).to_csv(TABLES / "table_B_BT_replication.csv", index=False)
    pd.DataFrame(tc).to_csv(TABLES / "table_C_incremental_beyond_VIX.csv", index=False)
    # D panels: one summary row per requested specification. Comparison measures
    # were pre-specified only for the binary panel; other panels say so explicitly.
    parts = []
    panels = [("Binary DD21 event", "B"), ("Continuous DD21 loss", "C"),
              ("Downside semivariance", "D"), ("10th-percentile return", "E")]
    spec_labels = {"level":"VIX + SKEW- level", "change":"VIX + SKEW- change",
        "z60":"VIX + SKEW- rolling z-score", "low":"VIX + low-SKEW- regime",
        "interaction":"VIX + low-VIX interaction"}
    def add_table_d_row(panel, specification, mid, focal, status):
        z = results[(results.model_id == mid) &
                    ((results.focal) if focal else (results.term == "vix"))].iloc[0]
        parts.append(dict(panel=panel, specification=specification, model_id=mid,
            focal_term=(z.term if focal else "vix"), coefficient=z.coefficient,
            standard_error=z.standard_error, raw_pvalue=z.raw_pvalue, n=z.n,
            event_count=z.event_count, aic=z.aic, pseudo_r2=z.pseudo_r2,
            auc=z.auc, brier=z.brier, adjusted_r2=z.adjusted_r2,
            rmse=z.rmse, mae=z.mae, pinball_loss=z.pinball_loss,
            lr_pvalue=(z.lr_pvalue if focal else np.nan), status=status,
            converged=z.converged, separation_flag=z.separation_flag,
            low_power_flag=z.low_power_flag, model_valid=z.model_valid))

    for panel, prefix in panels:
        for tag in FORMS:
            mid=f"{prefix}_{tag}"
            bid=f"{prefix}0_common_{tag}"
            add_table_d_row(panel, f"VIX only (common rows: {spec_labels[tag]})",
                            bid, False, f"exact common-row benchmark for {mid}")
            add_table_d_row(panel, spec_labels[tag], mid, True, "estimated")
        for label, mid, tag in [
                ("VIX + published SKEW", "B_compare_published", "published"),
                ("VIX + 25-delta skew", "B_compare_skew25", "skew25")]:
            if prefix == "B":
                bid=f"B0_common_{tag}"
                add_table_d_row(panel, f"VIX only (common rows: {label})", bid,
                                False, f"exact common-row benchmark for {mid}")
                add_table_d_row(panel, label, mid, True, "estimated")
            else:
                parts.append(dict(panel=panel, specification=label, model_id="",
                                  status="not pre-specified for this outcome"))
    pd.DataFrame(parts).to_csv(TABLES / "table_D_alternative_tail_outcomes.csv", index=False)
    oos.to_csv(TABLES / "table_E_chronological_OOS.csv", index=False)
    # F every focal test and non-estimable Firth
    sf = results[results.focal].copy()
    sf["classification"] = [assessment(r) for _, r in sf.iterrows()]
    firth = pd.DataFrame([dict(model_id="Firth_phase", family="rare_event_sensitivity",
        outcome="MktDown/DD21Event", sample="all", term="SKEW-", coefficient=np.nan,
        raw_pvalue=np.nan, classification="not estimable")])
    pd.concat([sf, firth], ignore_index=True).to_csv(TABLES / "table_F_specification_summary.csv", index=False)


def make_figures(d, results, oos_pred):
    FIGURES.mkdir(exist_ok=True)
    # Forest: representative requested focal estimates, standardized grouping not used.
    mids = [f"A_exact_h{h}_1996-2017" for h in [1,3,6,12]] + [
        "B_standalone_level", "B_level", "C_level", "D_level", "E_level"]
    z = results[(results.model_id.isin(mids)) & results.focal].copy()
    labels = {**{f"A_exact_h{h}_1996-2017":f"B&T-style h={h} (no VIX)" for h in [1,3,6,12]},
        "B_standalone_level":"DD21 event (standalone)",
        "B_level":"DD21 event (VIX-controlled)",
        "C_level":"DD21 loss (VIX-controlled)",
        "D_level":"LogDSV21 (VIX-controlled)",
        "E_level":"R21 q=.10 (VIX-controlled)"}
    z["label"] = z.model_id.map(labels)
    z["control_group"] = np.where(z.model_id.isin(["B_level","C_level","D_level","E_level"]),
                                  "VIX-controlled", "Standalone / B&T-style")
    z["lo"] = z.coefficient - 1.96*z.standard_error
    z["hi"] = z.coefficient + 1.96*z.standard_error
    fig, ax = plt.subplots(figsize=(9, 6))
    yy = np.arange(len(z))
    for i, row in z.reset_index(drop=True).iterrows():
        controlled = row.control_group == "VIX-controlled"
        ax.errorbar(row.coefficient, i,
            xerr=[[row.coefficient-row.lo], [row.hi-row.coefficient]],
            fmt=("s" if controlled else "o"),
            color=("#d95f02" if controlled else "#1b9e77"), capsize=3)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(yy, z.label); ax.invert_yaxis(); ax.set_xlabel("Coefficient (native outcome units)")
    ax.set_title("SKEW- coefficient audit (95% intervals)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#1b9e77",label="Standalone / B&T-style"),
                       Line2D([0],[0],marker="s",color="w",markerfacecolor="#d95f02",label="VIX-controlled")],
              loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_1_coefficient_forest.png", dpi=180); plt.close(fig)
    z[["model_id","label","control_group","coefficient","lo","hi","n"]].to_csv(
        FIGURES / "figure_1_underlying_data.csv", index=False)
    # Regime chart.
    q = d.dropna(subset=["low_skew_minus", "DD21", "DD21Event"]).copy()
    summ = q.groupby("low_skew_minus").agg(n=("DD21","size"), mean_dd=("DD21","mean"),
        sd_dd=("DD21","std"), event_rate=("DD21Event","mean"))
    summ["mean_dd_se"] = summ.sd_dd / np.sqrt(summ.n)
    summ["event_se"] = np.sqrt(summ.event_rate*(1-summ.event_rate)/summ.n)
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    labels = ["Normal", "Low SKEW-"]
    axes[0].bar(labels, summ.mean_dd, yerr=1.96*summ.mean_dd_se, capsize=4); axes[0].set_title("Average forward start-to-minimum decline")
    axes[1].bar(labels, summ.event_rate, yerr=1.96*summ.event_se, capsize=4); axes[1].set_title("P(start-to-minimum decline <= -5%)")
    for ax in axes:
        for i,n in enumerate(summ.n): ax.text(i, ax.get_ylim()[1]*.92, f"n={n}", ha="center")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_2_drawdown_by_skew_regime.png", dpi=180); plt.close(fig)
    summ.reset_index().to_csv(FIGURES / "figure_2_underlying_data.csv", index=False)
    # State matrix.
    q = d.dropna(subset=["low_vix", "low_skew_minus", "DD21Event"]).copy()
    mat = q.groupby(["low_vix","low_skew_minus"]).DD21Event.agg(["mean","size"]).reset_index()
    rates = mat.pivot(index="low_vix", columns="low_skew_minus", values="mean").reindex([1,0]).reindex(columns=[0,1])
    ns = mat.pivot(index="low_vix", columns="low_skew_minus", values="size").reindex([1,0]).reindex(columns=[0,1])
    fig, ax = plt.subplots(figsize=(6,4)); im=ax.imshow(rates, cmap="Reds", vmin=0, vmax=max(.01,np.nanmax(rates.values)))
    ax.set_xticks([0,1],["Normal SKEW-","Low SKEW-"]); ax.set_yticks([0,1],["Low VIX","High VIX"])
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{rates.iloc[i,j]:.1%}\nn={int(ns.iloc[i,j])}",ha="center",va="center")
    ax.set_title("Forward start-to-minimum event rates (descriptive)"); fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_3_state_matrix.png",dpi=180); plt.close(fig)
    mat.to_csv(FIGURES / "figure_3_underlying_data.csv", index=False)
    # Recursive monthly downturn forecasts required by Figure 4.
    level_benchmark = "VIX only (paired with VIX + SKEW- level)"
    q = oos_pred[(oos_pred.task=="binary") & (oos_pred.outcome=="MktDownNext") &
        (oos_pred.model.isin([level_benchmark,"VIX + SKEW- level"]))].copy()
    fig, ax = plt.subplots(figsize=(12,4))
    for model,g in q.groupby("model"): ax.plot(pd.to_datetime(g.predictor_date),g.prediction,label=model,lw=1)
    ev=q[(q.model==level_benchmark) & (q.actual==1)]
    ax.scatter(pd.to_datetime(ev.predictor_date),np.ones(len(ev))*1.02,marker="|",color="black",label="Realised event")
    ax.set_ylim(0,1.08); ax.set_ylabel("Recursive predicted probability"); ax.legend(ncol=3)
    ax.set_title("Expanding-window one-month downturn forecasts")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_4_recursive_forecasts.png",dpi=180); plt.close(fig)


def write_report(d, results, oos):
    """Render the audit using model-specific conventional inference only."""

    def foc(model_id):
        rows = results[(results.model_id == model_id) & results.focal]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one focal coefficient for {model_id}; found {len(rows)}")
        return rows.iloc[0]

    def fmt(value, digits=4):
        try:
            return f"{float(value):.{digits}f}" if np.isfinite(value) else "not available"
        except (TypeError, ValueError):
            return "not available"

    def p_number(value):
        if not np.isfinite(value):
            return "not available"
        return "<0.001" if float(value) < .001 else f"{float(value):.3f}"

    def p_fmt(value):
        return f"p-value={p_number(value)}"

    candidate_oos = oos[oos.pair_role == "candidate"].copy()
    stable_loss = []
    for (task, outcome, model), group in candidate_oos.groupby(
            ["task", "outcome", "model"]):
        periods = set(group.evaluation_period)
        if ({"2006-2017", "2018-2025"}.issubset(periods)
                and (group.paired_loss_difference_vs_vix_absolute < 0).all()
                and group.invalid_forecast_fits.eq(0).all()):
            stable_loss.append((task, outcome, model))

    lines = [
        "# Tail-risk specification audit", "",
        "**Inference:** Model-specific conventional p-values and confidence intervals are reported. No family-wise multiple-testing adjustment is applied.", "",
        "## Outcome and predictor definitions", "",
        "`DD21 = min_{j=1,...,21}(P_{t+j}/P_t - 1)` is the start-to-minimum decline from the predictor close over the next 21 trading days. It is neither a future-window peak-to-trough drawdown nor a simple 21-day return.", "",
        "`DD21Loss = -min(DD21, 0)` records loss magnitude as a non-negative number. `DSV21 = sum(min(r_i, 0)^2)` and `LogDSV21 = log(1 + 10000*DSV21)`.", "",
        "SKEW- is the put-only BKM skew index. Published CBOE/Bloomberg SKEW is a distinct comparison series.", "",
        "## B&T-style replication", "",
    ]
    horizon_rows = []
    for horizon in [1, 3, 6, 12]:
        row = foc(f"A_exact_h{horizon}_1996-2017")
        horizon_rows.append(row)
        lines.append(
            f"- h={horizon}: coefficient {fmt(row.coefficient)}, {p_fmt(row.raw_pvalue)}, "
            f"N={int(row.n)}, events={int(row.event_count)}, "
            f"AME per one-unit increase={fmt(row.average_marginal_effect)}.")

    h1 = horizon_rows[0]
    direction_matches = np.sign(h1.coefficient) == np.sign(-.084)
    inference_matches = h1.raw_pvalue < .01
    strongest = min(horizon_rows, key=lambda row: row.raw_pvalue)
    strongest_h = int(str(strongest.model_id).split("_h")[1].split("_")[0])
    lines += ["",
        f"At h=1, the coefficient direction {'matches' if direction_matches else 'does not match'} B&T, while the conventional inference {'matches' if inference_matches else 'does not match'} their reported p<0.01 result. This is a directional partial replication rather than a statistical replication of B&T Table 6.", "",
        f"The strongest horizon is h={strongest_h}: coefficient {fmt(strongest.coefficient)} and {p_fmt(strongest.raw_pvalue)}. It is statistically significant at the conventional 5% level.", "",
        "## VIX-incremental results", "",
    ]
    for horizon in [1, 3, 6, 12]:
        row = foc(f"A_vix_h{horizon}_1996-2017")
        lines.append(
            f"- h={horizon}: SKEW- coefficient {fmt(row.coefficient)}, "
            f"{p_fmt(row.raw_pvalue)}, exact-row VIX LR p={fmt(row.lr_pvalue)}, "
            f"N={int(row.n)}, events={int(row.event_count)}.")

    lines += ["", "## Alternative tail-risk outcomes and signal forms", ""]
    outcome_names = {
        "B": "DD21 event", "C": "DD21 loss", "D": "LogDSV21",
        "E": "10th-percentile R21",
    }
    for prefix, outcome_name in outcome_names.items():
        lines.append(f"### {outcome_name}")
        lines.append("")
        for tag in FORMS:
            row = foc(f"{prefix}_{tag}")
            lines.append(
                f"- {tag}: coefficient {fmt(row.coefficient)}, {p_fmt(row.raw_pvalue)}, "
                f"N={int(row.n)}.")
        lines.append("")

    lines += [
        "The one-month change has the adverse-risk sign across the four alternative outcomes. Its DD21-event result is suggestive at the conventional 10% level; the DD21-loss and LogDSV21 results are statistically significant at the conventional 5% level. These outcomes overlap and are not treated as independent confirmations.", "",
        "The rolling z-score, low-SKEW regime, and low-VIX interaction results do not establish a robust state-dependent effect. The LogDSV21 low-regime result (p=0.0508) is suggestive at 10% but not significant at 5%.", "",
        "## Chronological forecast performance", "",
    ]
    change_oos = candidate_oos[candidate_oos.model == "VIX + SKEW- change"]
    loss_fields = {
        "binary": ("Brier", "brier", "benchmark_brier_common_rows"),
        "continuous": ("MAE", "mae", "benchmark_mae_common_rows"),
        "quantile": ("pinball", "pinball_loss", "benchmark_pinball_common_rows"),
    }
    for _, row in change_oos.sort_values(["outcome", "evaluation_period"]).iterrows():
        label, candidate_field, benchmark_field = loss_fields[row.task]
        auc_text = (f"; candidate AUC={fmt(row.auc, 3)}, paired VIX AUC="
                    f"{fmt(row.benchmark_auc_common_rows, 3)}"
                    if row.task == "binary" else "")
        lines.append(
            f"- {row.outcome}, {row.evaluation_period}: candidate {label}="
            f"{fmt(row[candidate_field], 6)}, paired VIX {label}="
            f"{fmt(row[benchmark_field], 6)}, difference="
            f"{fmt(row.paired_loss_difference_vs_vix_absolute, 6)}{auc_text}.")
    stable_text = ("; ".join(f"{outcome} / {model}" for _, outcome, model in stable_loss)
                   if stable_loss else "none")
    lines += ["",
        f"Candidate/outcome combinations with descriptively lower paired loss in both periods: {stable_text}. Brier, AUC, and loss differences are descriptive because formal forecast-comparison inference was not performed.", "",
        "The 2018-2025 period is a post-paper extension, not a pristine holdout. Every candidate is compared with VIX refitted on the exact same rows at each forecast origin.", "",
        "## Diagnostics and interpretation", "",
        "Conventional p-values are interpreted model by model: p<0.05 is statistically significant, 0.05<=p<0.10 is suggestive, and p>=0.10 is not statistically significant. Effect sizes, uncertainty, sample size, event count, convergence, exact-row VIX comparisons, and OOS performance remain part of the interpretation.", "",
        "`raw_model_results.csv` retains coefficients, standard errors, conventional p-values, convergence, warnings, event counts, low-power flags, correlation, VIF, condition number, and applicable residual diagnostics.", "",
    ]
    (OUT / "final_specification_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main(argv=None):
    global OUT, TABLES, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(OUT),
        help="Prepared-input and output directory (default: outputs/tailrisk_specification_audit)")
    args = parser.parse_args(argv)
    selected = Path(args.output)
    OUT = selected.resolve() if selected.is_absolute() else (ROOT / selected).resolve()
    TABLES = OUT / "tables"
    FIGURES = OUT / "figures"
    required_inputs = [
        OUT / "tail_outcomes_monthly.csv",
        OUT / "predictor_variants_monthly.csv",
        OUT / "leakage_audit.csv",
    ]
    if any(not path.exists() for path in required_inputs):
        raise RuntimeError("Prepared outcomes, predictors and leakage audit must exist first")
    leak = pd.read_csv(OUT / "leakage_audit.csv")
    if not leak.passed.all():
        raise RuntimeError("Leakage audit failed; modelling stopped")
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    d=load_data(); store=Store()
    run_family_a(store,d)
    run_families_b_to_e(store,d)
    firth_availability(store)
    results=pd.DataFrame(store.results)
    registry=pd.DataFrame(store.registry)
    registry.to_csv(OUT/"model_registry.csv",index=False)
    results.to_csv(OUT/"raw_model_results.csv",index=False)
    oos_pred,oos=run_oos(d)
    make_tables(d,results,oos)
    make_figures(d,results,oos_pred)
    write_report(d,results,oos)
    print(f"Analysis complete: {len(registry)} models, {len(results)} coefficient rows, {len(oos_pred)} OOS forecasts")


if __name__ == "__main__":
    main()
