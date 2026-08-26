#!/usr/bin/env python3
"""Build the 1996-2017 BKM moments and put-side SKEW− series."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_skew_minus import (  # noqa: E402
    OPTION_COLUMNS,
    assert_chunk_date_order,
    calculate_expiry,
    load_rates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=Path("data/raw/new raw.zip"))
    parser.add_argument(
        "--rates", type=Path,
        default=Path("data/raw/riskfree/mwllinpfidkhuyz2.csv"),
    )
    parser.add_argument(
        "--pairs", type=Path,
        default=Path(
            "outputs/validation/bt_expiry_rule_replication/strict_2_90/"
            "part3_strict_expiry_pairs.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/validation/bt_expiry_rule_replication/strict_2_90"),
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def calculate_strict_date(
    frame: pd.DataFrame,
    trade_date: pd.Timestamp,
    near_expiry: pd.Timestamp,
    far_expiry: pd.Timestamp,
    curves,
    curve_dates,
) -> tuple[dict, list[dict]]:
    diagnostics = []
    results = []
    for expiry in [near_expiry, far_expiry]:
        chain = frame[frame["exdate"].eq(expiry)]
        result, diagnostic = calculate_expiry(
            chain, trade_date, expiry, curves, curve_dates
        )
        diagnostics.append(diagnostic)
        results.append(result)
    if not all(results):
        return {
            "date": trade_date, "status": "failed",
            "reason": "near_or_far_expiry_failed",
            "near_expiry": near_expiry, "far_expiry": far_expiry,
        }, diagnostics
    near, far = results
    if far["dte"] == near["dte"]:
        raise ValueError(f"Equal maturity pair on {trade_date.date()}")
    weight_near = (far["dte"] - 30.0) / (far["dte"] - near["dte"])
    minus_30 = weight_near * near["minus"]["skew"] + (1.0 - weight_near) * far["minus"]["skew"]
    return {
        "date": trade_date,
        "status": "ok",
        "reason": "",
        "near_expiry": near_expiry,
        "far_expiry": far_expiry,
        "near_dte": near["dte"],
        "far_dte": far["dte"],
        "weight_near": weight_near,
        "interpolation_kind": (
            "interpolation" if 0 < weight_near < 1 else
            "endpoint" if weight_near in {0, 1} else "extrapolation"
        ),
        "skew_minus_raw": minus_30,
        "skew_minus_index": 100.0 - 10.0 * minus_30,
    }, diagnostics


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(
        args.pairs,
        parse_dates=["trade_date", "selected_near_expiry", "selected_far_expiry"],
    )
    if not pairs["required_pair_available_yes_no"].all():
        raise ValueError("Strict expiry gate has not passed for every trade date")
    selected_by_date = {
        row.trade_date.strftime("%Y-%m-%d"): (
            row.selected_near_expiry.strftime("%Y-%m-%d"),
            row.selected_far_expiry.strftime("%Y-%m-%d"),
        )
        for row in pairs.itertuples()
    }
    selected_keys = {
        f"{date}|{expiry}"
        for date, expiries in selected_by_date.items()
        for expiry in expiries
    }
    curves, curve_dates = load_rates(args.rates)
    daily_rows = []
    expiry_rows = []
    carry = pd.DataFrame()
    raw_rows = selected_rows = 0
    previous_last_date: pd.Timestamp | None = None

    reader = pd.read_csv(
        args.options, usecols=OPTION_COLUMNS, chunksize=args.chunksize,
        low_memory=False,
    )
    for number, chunk in enumerate(reader, 1):
        raw_rows += len(chunk)
        previous_last_date = assert_chunk_date_order(
            chunk, number, previous_last_date
        )
        last_date = chunk["date"].iloc[-1]
        keys = chunk["date"].astype(str) + "|" + chunk["exdate"].astype(str)
        chosen = chunk[
            chunk["am_settlement"].eq(1)
            & keys.isin(selected_keys)
        ].copy()
        selected_rows += len(chosen)
        combined = pd.concat([carry, chosen], ignore_index=True) if len(carry) else chosen
        complete = combined[~combined["date"].eq(last_date)]
        carry = combined[combined["date"].eq(last_date)].copy()
        for date_text, day in complete.groupby("date", sort=False):
            near_text, far_text = selected_by_date[str(date_text)]
            trade_date = pd.Timestamp(date_text)
            day["exdate"] = pd.to_datetime(day["exdate"], errors="raise")
            result, diagnostics = calculate_strict_date(
                day, trade_date, pd.Timestamp(near_text), pd.Timestamp(far_text),
                curves, curve_dates,
            )
            daily_rows.append(result)
            expiry_rows.extend(diagnostics)
        if number % 5 == 0:
            print(
                f"Read {raw_rows:,} rows; constructed {len(daily_rows):,} dates",
                flush=True,
            )
    for date_text, day in carry.groupby("date", sort=False):
        near_text, far_text = selected_by_date[str(date_text)]
        trade_date = pd.Timestamp(date_text)
        day["exdate"] = pd.to_datetime(day["exdate"], errors="raise")
        result, diagnostics = calculate_strict_date(
            day, trade_date, pd.Timestamp(near_text), pd.Timestamp(far_text),
            curves, curve_dates,
        )
        daily_rows.append(result)
        expiry_rows.extend(diagnostics)

    daily = pd.DataFrame(daily_rows).sort_values("date")
    expiry = pd.DataFrame(expiry_rows).sort_values(["date", "exdate"])
    if len(daily) != len(pairs):
        missing_dates = sorted(set(pairs["trade_date"]) - set(daily["date"]))
        raise ValueError(
            f"Constructed {len(daily)} of {len(pairs)} dates; missing {missing_dates[:10]}"
        )
    daily.to_csv(args.output / "part5_daily_strict_skew.csv", index=False)
    expiry.to_csv(args.output / "part5_strict_expiry_diagnostics.csv", index=False)
    ok = daily[daily["status"].eq("ok")]
    ok_expiry = expiry[expiry["status"].eq("ok")]
    audit = {
        "options_input": str(args.options.resolve()),
        "rates_input": str(args.rates.resolve()),
        "expiry_pairs_input": str(args.pairs.resolve()),
        "raw_rows_read": raw_rows,
        "strict_selected_option_rows": selected_rows,
        "daily_dates": len(daily),
        "successful_daily_dates": len(ok),
        "failed_daily_dates": len(daily) - len(ok),
        "daily_failure_reasons": daily.loc[
            daily["status"].ne("ok"), "reason"
        ].value_counts().to_dict(),
        "expiry_failure_reasons": expiry.loc[
            expiry["status"].ne("ok"), "reason"
        ].value_counts().to_dict(),
        "successful_date_min": str(ok["date"].min().date()),
        "successful_date_max": str(ok["date"].max().date()),
        "asof_rate_expiries": int(ok_expiry["curve_lag_days"].gt(0).sum()),
        "rate_endpoint_expiries": int(ok_expiry["rate_outside_curve"].sum()),
        "raw_crossed_quotes_in_selected_expiries": int(
            ok_expiry["raw_crossed_quotes"].sum()
        ),
        "selected_expiries_with_crossed_quotes": int(
            ok_expiry["raw_crossed_quotes"].gt(0).sum()
        ),
        "selected_dates_with_crossed_quotes": int(
            ok_expiry.loc[ok_expiry["raw_crossed_quotes"].gt(0), "date"].nunique()
        ),
        "crossed_quote_treatment": "dropped before midpoint construction",
        "chunk_date_order_assertions_passed": True,
        "interpolated_30d_dates": int(ok["interpolation_kind"].eq("interpolation").sum()),
        "exact_30d_endpoint_dates": int(ok["interpolation_kind"].eq("endpoint").sum()),
        "extrapolated_30d_dates": int(ok["interpolation_kind"].eq("extrapolation").sum()),
        "construction_method": {
            "bkm_equations": "unchanged from scripts/build_skew_minus.py",
            "zero_bid_filter": "unchanged",
            "deltaK": "unchanged",
            "rates": "unchanged OptionMetrics zero curve",
            "interpolation": "unchanged direct linear interpolation of expiry S",
            "strike_scale": "strike_price/1000",
        },
    }
    (args.output / "part5_construction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote strict daily series: {len(ok):,}/{len(daily):,} successful dates",
        flush=True,
    )


if __name__ == "__main__":
    main()
