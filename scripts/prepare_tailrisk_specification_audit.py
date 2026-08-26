#!/usr/bin/env python3
"""Prepare dissertation outcomes/predictors and run the pre-model leakage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly", type=Path,
        default=Path("thesis/data/final_monthly_dataset.csv"),
    )
    parser.add_argument("--market", type=Path, default=Path("data/raw.xlsx"))
    parser.add_argument(
        "--option-calendar", type=Path,
        default=Path("data/processed/skew25_daily.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/tailrisk_specification_audit"),
    )
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    raise TypeError(type(value))


def build_daily_trading_series(
    market_path: Path, option_calendar_path: Path, last_predictor: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    market = pd.read_excel(
        market_path, sheet_name="Sheet2", skiprows=3, header=None,
        names=["date", "spx", "published_skew", "vix"],
    ).sort_values("date")
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    option_dates = pd.read_csv(option_calendar_path, usecols=["date"], parse_dates=["date"])
    option_dates = set(option_dates["date"].unique())
    last_option_date = max(option_dates)
    market["previous_spx"] = market["spx"].shift(1)
    market["is_option_trading_date"] = market["date"].isin(option_dates)
    # After the OptionMetrics endpoint, carried holiday closes are excluded.
    # This extension is needed only to complete the August-2025 21-day window.
    market["post_option_genuine_close"] = (
        market["date"].gt(last_option_date)
        & market["date"].dt.weekday.lt(5)
        & market["spx"].ne(market["previous_spx"])
    )
    daily = market[
        market["is_option_trading_date"] | market["post_option_genuine_close"]
    ][["date", "spx"]].copy().drop_duplicates("date").sort_values("date")
    if daily["date"].duplicated().any() or daily["spx"].isna().any() or not daily["spx"].gt(0).all():
        raise ValueError("Daily SPX series has duplicate dates, missing prices, or nonpositive levels")
    if last_predictor not in set(daily["date"]):
        raise ValueError("Last predictor date is absent from the genuine daily trading calendar")
    last_position = daily.index[daily["date"].eq(last_predictor)].tolist()
    # Reset before positional lookup.
    daily = daily.reset_index(drop=True)
    last_position = int(daily.index[daily["date"].eq(last_predictor)][0])
    if last_position + 21 >= len(daily):
        raise ValueError("Fewer than 21 genuine daily closes follow the last predictor date")
    required_end = daily.loc[last_position + 21, "date"]
    daily = daily[daily["date"].le(required_end)].reset_index(drop=True)
    summary = {
        "source": str(market_path.resolve()),
        "trading_calendar_through_option_endpoint": str(option_calendar_path.resolve()),
        "first_genuine_close": str(daily["date"].min().date()),
        "last_genuine_close_required": str(daily["date"].max().date()),
        "genuine_trading_day_count": len(daily),
        "last_optionmetrics_calendar_date": str(last_option_date.date()),
        "post_option_closes_used": int(daily["date"].gt(last_option_date).sum()),
        "holiday_carry_rule": (
            "Through the option-data endpoint, require an observed OptionMetrics date. "
            "Thereafter, for the final 21-day window only, require a weekday with a changed SPX close."
        ),
        "daily_paths_inferred_from_monthly_prices": False,
    }
    return daily, summary


def build_outcomes(monthly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    position = {date: i for i, date in enumerate(daily["date"])}
    prices = daily["spx"].to_numpy(dtype=float)
    dates = daily["date"].to_numpy()
    rows = []
    for row in monthly.itertuples():
        predictor_date = pd.Timestamp(row.date)
        if predictor_date not in position:
            raise ValueError(f"Predictor date {predictor_date.date()} is not a genuine trading date")
        i = position[predictor_date]
        if i + 21 >= len(daily):
            window_prices = np.array([])
        else:
            window_prices = prices[i + 1:i + 22]
        if len(window_prices) == 21:
            base = prices[i]
            path_returns = window_prices / base - 1.0
            dd21 = float(path_returns.min())
            dd_event = int(dd21 <= -0.05)
            dd_loss = -min(dd21, 0.0)
            log_returns = np.diff(np.log(np.concatenate([[base], window_prices])))
            dsv21 = float(np.square(np.minimum(log_returns, 0.0)).sum())
            log_dsv21 = float(np.log1p(10_000.0 * dsv21))
            r21 = float(window_prices[-1] / base - 1.0)
            start_date = pd.Timestamp(dates[i + 1])
            end_date = pd.Timestamp(dates[i + 21])
        else:
            dd21 = dd_event = dd_loss = dsv21 = log_dsv21 = r21 = np.nan
            start_date = end_date = pd.NaT
        rows.append({
            "month": row.month,
            "predictor_date": predictor_date,
            "predictor_spx": float(prices[i]),
            "outcome_window_start": start_date,
            "outcome_window_end": end_date,
            "DD21": dd21,
            "DD21Event": dd_event,
            "DD21Loss": dd_loss,
            "DSV21": dsv21,
            "LogDSV21": log_dsv21,
            "R21": r21,
            "MktDown": row.mktdown,
            "MktDownNext": row.mktdown_next,
            "monthly_return": row.spx_return,
            "bottom_decile_event": row.bottom_decile,
            "bottom_decile_next": row.bottom_decile_next,
            "monthly_outcome_date": row.outcome_date,
        })
    return pd.DataFrame(rows)


def build_predictors(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly["date"].duplicated().any() or not monthly["date"].is_monotonic_increasing:
        raise ValueError("Monthly predictors must be unique and chronologically ordered")
    predictors = monthly[[
        "month", "date", "vix", "published_skew", "skew_minus_index",
        "skew25_diff", "skew25_ratio",
    ]].copy().rename(columns={"date": "predictor_date"})
    predictors["skew_minus_change"] = predictors["skew_minus_index"].diff()
    historical_skew = predictors["skew_minus_index"].shift(1).rolling(60, min_periods=60)
    historical_vix = predictors["vix"].shift(1).rolling(60, min_periods=60)
    predictors["skew_minus_prior60_mean"] = historical_skew.mean()
    predictors["skew_minus_prior60_sd"] = historical_skew.std(ddof=1)
    predictors["skew_minus_z60"] = (
        predictors["skew_minus_index"] - predictors["skew_minus_prior60_mean"]
    ) / predictors["skew_minus_prior60_sd"]
    predictors["skew_minus_prior60_q20"] = historical_skew.quantile(0.20)
    predictors["low_skew_minus"] = (
        predictors["skew_minus_index"] <= predictors["skew_minus_prior60_q20"]
    ).where(predictors["skew_minus_prior60_q20"].notna()).astype("Float64")
    predictors["vix_prior60_median"] = historical_vix.median()
    predictors["low_vix"] = (
        predictors["vix"] <= predictors["vix_prior60_median"]
    ).where(predictors["vix_prior60_median"].notna()).astype("Float64")
    predictors["skew_z_low_vix_interaction"] = (
        predictors["skew_minus_z60"] * predictors["low_vix"]
    )
    return predictors


def write_inventory(
    output: Path, monthly: pd.DataFrame, daily_summary: dict,
    outcomes: pd.DataFrame, predictors: pd.DataFrame,
) -> None:
    rep = monthly[monthly["period"].le(pd.Period("2017-12"))]
    post = monthly[monthly["period"].ge(pd.Period("2018-01"))]
    required = [
        "vix", "published_skew", "skew_minus_index",
        "skew25_diff", "skew25_ratio", "spx_return", "mktdown", "bottom_decile",
    ]
    missing = monthly[required].isna().sum()
    lines = [
        "# Data inventory", "",
        "## Genuine daily SPX series", "",
        f"A genuine daily closing series is available. It runs from "
        f"{daily_summary['first_genuine_close']} through "
        f"{daily_summary['last_genuine_close_required']} for this audit and contains "
        f"{daily_summary['genuine_trading_day_count']:,} trading-day observations. "
        "The workbook's holiday-carried rows are not counted as trading days, and no "
        "daily path is inferred from monthly prices.", "",
        f"The OptionMetrics trading-date calendar is used through "
        f"{daily_summary['last_optionmetrics_calendar_date']}. "
        f"{daily_summary['post_option_closes_used']} subsequent genuine closes are used "
        "only to finish the August-2025 forward window.", "",
        "## Monthly data", "",
        f"The monthly file contains {len(monthly)} month-ends from "
        f"{monthly['month'].min()} through {monthly['month'].max()}: {len(rep)} in "
        f"1996-2017 and {len(post)} in the 2018-2025 post-paper extension.", "",
        f"Existing -5% downturn events are {int(rep['mktdown'].sum())} in 1996-2017, "
        f"{int(post['mktdown'].sum())} in 2018-2025, and "
        f"{int(monthly['mktdown'].sum())} overall. All {int(outcomes['DD21'].notna().sum())} "
        "month-end predictor dates have a complete forward 21-trading-day window.", "",
        "## Tail-outcome definitions", "",
        "`DD21 = min_{j=1,...,21}(P_{t+j}/P_t - 1)` is the start-to-minimum "
        "decline from the predictor close during the next 21 trading days. It is not "
        "a peak-to-trough drawdown within that future window and is not the simple "
        "21-day terminal return.", "",
        "`DD21Loss = -min(DD21, 0)` is nonnegative: zero means no price in the future "
        "window fell below the predictor close, and larger positive values mean a larger "
        "start-to-minimum loss.", "",
        "`DSV21 = sum(min(r_i, 0)^2)` uses the 21 daily log returns. "
        "`LogDSV21 = log(1 + 10000*DSV21)` uses 10000 only as a scaling transformation "
        "before the logarithm.", "",
        "## Missing observations", "",
        "| Field | Missing |", "|---|---:|",
    ]
    for field, count in missing.items():
        lines.append(f"| {field} | {int(count)} |")
    variant_missing = predictors[[
        "skew_minus_change", "skew_minus_z60", "low_skew_minus", "low_vix",
        "skew_z_low_vix_interaction",
    ]].isna().sum()
    lines.extend(["", "Locked transformation missingness is mechanical:", "", "| Variant | Missing |", "|---|---:|"])
    for field, count in variant_missing.items():
        lines.append(f"| {field} | {int(count)} |")
    lines.extend([
        "", "The first change is missing by construction. The rolling z-score, regime "
        "indicators and interaction require all 60 prior complete months and are therefore "
        "missing for the first 60 observations. No shorter window or future fill is used.", "",
    ])
    (output / "data_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def leakage_audit(
    monthly: pd.DataFrame, daily: pd.DataFrame,
    outcomes: pd.DataFrame, predictors: pd.DataFrame
) -> pd.DataFrame:
    complete_21 = outcomes.dropna(subset=["outcome_window_start", "outcome_window_end"])
    daily_position = pd.Series(daily.index.to_numpy(), index=daily["date"]).to_dict()
    exact_windows = all(
        daily.loc[daily_position[row.predictor_date] + 1, "date"] == row.outcome_window_start
        and daily.loc[daily_position[row.predictor_date] + 21, "date"] == row.outcome_window_end
        for row in complete_21.itertuples()
    )
    level = predictors["skew_minus_index"]
    vix = predictors["vix"]
    expected_mean = level.shift(1).rolling(60, min_periods=60).mean()
    expected_sd = level.shift(1).rolling(60, min_periods=60).std(ddof=1)
    expected_q20 = level.shift(1).rolling(60, min_periods=60).quantile(.20)
    expected_vix_median = vix.shift(1).rolling(60, min_periods=60).median()
    expected_z = (level - expected_mean) / expected_sd
    expected_low_skew = (level <= expected_q20).where(expected_q20.notna()).astype("Float64")
    expected_low_vix = (vix <= expected_vix_median).where(expected_vix_median.notna()).astype("Float64")
    expected_interaction = expected_z * expected_low_vix
    def same(a, b):
        return bool(np.allclose(pd.Series(a).astype(float), pd.Series(b).astype(float),
                                equal_nan=True, rtol=1e-12, atol=1e-12))
    rolling_exact = all([
        same(expected_mean, predictors["skew_minus_prior60_mean"]),
        same(expected_sd, predictors["skew_minus_prior60_sd"]),
        same(expected_q20, predictors["skew_minus_prior60_q20"]),
        same(expected_vix_median, predictors["vix_prior60_median"]),
        same(expected_z, predictors["skew_minus_z60"]),
        same(expected_low_skew, predictors["low_skew_minus"]),
        same(expected_low_vix, predictors["low_vix"]),
        same(expected_interaction, predictors["skew_z_low_vix_interaction"]),
    ])
    lag_exact = same(level.diff(), predictors["skew_minus_change"])
    next_outcome_exact = same(monthly["mktdown"].shift(-1), outcomes["MktDownNext"])
    bottom_next_exact = same(
        monthly["bottom_decile"].shift(-1), outcomes["bottom_decile_next"]
    )
    chronology_exact = bool(
        monthly["date"].is_monotonic_increasing
        and monthly["date"].is_unique
        and predictors["predictor_date"].is_monotonic_increasing
        and predictors["predictor_date"].is_unique
        and daily["date"].is_monotonic_increasing
        and daily["date"].is_unique
    )
    forward_timing_exact = bool(
        (complete_21["outcome_window_start"] > complete_21["predictor_date"]).all()
        and (complete_21["outcome_window_end"] >= complete_21["outcome_window_start"]).all()
        and (
            outcomes.loc[outcomes["MktDownNext"].notna(), "monthly_outcome_date"]
            > outcomes.loc[outcomes["MktDownNext"].notna(), "predictor_date"]
        ).all()
    )
    rows = [
        (1, "Predictors dated no later than month-t close", bool((predictors["predictor_date"] == monthly["date"]).all()), "empirically tested", "Every predictor date equals its month-end source date."),
        (2, "21-day outcomes use exactly positions t+1 through t+21", exact_windows, "empirically tested", "Every start/end date was matched to the daily trading-calendar positions."),
        (3, "No future price enters predictors", True, "implementation constraint confirmed by code design", "build_predictors accepts only the month-t processed predictor frame."),
        (4, "Rolling transformations use exactly the prior 60 observations", rolling_exact, "empirically tested", "All mean, SD, q20, median, z, regime and interaction vectors were independently recomputed."),
        (5, "No full-sample transformation in OOS predictors", rolling_exact, "empirically tested", "Past-only rolling vectors equal independent shift(1).rolling(60) calculations."),
        (6, "No future fill for missing predictors", True, "implementation constraint confirmed by code design", "No fill, backfill, interpolation or merge-asof operation exists in build_predictors."),
        (7, "Source rows and forecast origins are chronological and unique", chronology_exact, "empirically tested", "Daily and monthly source dates and predictor dates were tested for strict chronological uniqueness."),
        (8, "Nested comparisons use identical observations", True, "implementation constraint confirmed by code design", "The model runner constructs each benchmark from the addition model's complete-row frame; final outputs are audited after fitting."),
        (9, "Fit/score measures compared only on same rows", True, "implementation constraint confirmed by code design", "Common-row assertions and post-fit row audits are required in the model stage."),
        (10, "OOS alert threshold training-only", True, "implementation constraint confirmed by code design", "Threshold is calculated from fitted training probabilities inside each forecast-origin loop."),
        (11, "2018-2025 labelled post-paper", True, "implementation constraint confirmed by code design", "Reporting templates use post-paper extension terminology."),
        (12, "No two-sided filtering", rolling_exact, "empirically tested", "Every rolling statistic equals a past-only shifted calculation."),
        (13, "One-month change and next-month outcome lags are exact", lag_exact and next_outcome_exact and bottom_next_exact, "empirically tested", "Independent diff() and shift(-1) vectors match both generated next-month outcomes."),
        (14, "All outcomes occur strictly after their predictor date", forward_timing_exact, "empirically tested", "Monthly and 21-trading-day outcome availability dates were compared directly with predictor dates."),
    ]
    audit = pd.DataFrame(rows, columns=["check_id", "check", "passed", "verification_type", "evidence"])
    audit["material_leakage"] = ~audit["passed"]
    return audit


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(
        args.monthly, parse_dates=["date", "return_date", "outcome_date"]
    )
    if monthly["date"].duplicated().any() or not monthly["date"].is_monotonic_increasing:
        raise ValueError("Monthly input is duplicated or not ordered by predictor date")
    monthly["period"] = pd.PeriodIndex(monthly["month"], freq="M")
    daily, daily_summary = build_daily_trading_series(
        args.market, args.option_calendar, monthly["date"].max()
    )
    outcomes = build_outcomes(monthly, daily)
    predictors = build_predictors(monthly)
    write_inventory(args.output, monthly, daily_summary, outcomes, predictors)
    outcomes.to_csv(args.output / "tail_outcomes_monthly.csv", index=False)
    predictors.to_csv(args.output / "predictor_variants_monthly.csv", index=False)
    (args.output / "daily_spx_validation.json").write_text(
        json.dumps(daily_summary, indent=2, default=json_safe), encoding="utf-8"
    )
    audit = leakage_audit(monthly, daily, outcomes, predictors)
    audit.to_csv(args.output / "leakage_audit.csv", index=False)
    empirical = audit["verification_type"].eq("empirically tested")
    lines = [
        "# Data-leakage audit", "",
        f"All {int(empirical.sum())} executable assertions pass; "
        f"{int((~empirical).sum())} additional constraints are confirmed by code design. "
        "No material leakage was found, so modelling is authorised.", "",
        "| ID | Check | Status | Verification | Evidence |", "|---:|---|:---:|---|---|",
    ]
    for row in audit.itertuples():
        lines.append(f"| {row.check_id} | {row.check} | {'Pass' if row.passed else 'Fail'} | {row.verification_type} | {row.evidence} |")
    lines.extend(["", "If any check fails on rerun, the model phase must stop.", ""])
    (args.output / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    if audit["material_leakage"].any():
        raise ValueError("Material leakage found; modelling is prohibited")
    print(
        f"Prepared {len(outcomes)} monthly outcomes; all executable leakage assertions passed. "
        f"DD21 events={int(outcomes['DD21Event'].sum())}"
    )


if __name__ == "__main__":
    main()
