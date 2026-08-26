#!/usr/bin/env python3
"""Construct the standardized 30-day 25-delta SPX IV skew from IvyDB."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Explicit failure-marker screen used defensively in addition to numeric,
# finite and positivity checks. The supplied extract contains none of them;
# observed failed values are missing (NaN).
IV_SENTINELS = {-9999.0, -999.0, -99.99, 99.99, 999.0, 9999.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/25deltafull.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/skew25_daily.csv"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/25d/audit.json"))
    parser.add_argument(
        "--validity-audit", type=Path,
        default=Path("outputs/25d/iv25_validity_audit.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.input) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {args.input}, found {members}")
        with archive.open(members[0]) as source:
            raw = pd.read_csv(source)

    required = {"secid", "date", "days", "delta", "impl_volatility", "dispersion", "cp_flag", "index_flag"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    raw_columns = list(raw.columns)
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw["impl_volatility_original"] = raw["impl_volatility"]
    raw["impl_volatility"] = pd.to_numeric(raw["impl_volatility"], errors="coerce")
    key_duplicates = int(raw.duplicated(["secid", "date", "days", "delta", "cp_flag"]).sum())

    selected = raw[
        raw["secid"].eq(108105)
        & raw["index_flag"].eq(1)
        & raw["days"].eq(30)
        & (((raw["cp_flag"] == "P") & raw["delta"].eq(-25))
           | ((raw["cp_flag"] == "C") & raw["delta"].eq(25)))
    ].copy()
    original = selected["impl_volatility_original"]
    numeric = selected["impl_volatility"]
    reason_masks = {
        "missing_original": original.isna(),
        "non_numeric": original.notna() & numeric.isna(),
        "non_finite": numeric.notna() & ~np.isfinite(numeric),
        "known_failure_sentinel": numeric.isin(IV_SENTINELS),
        "nonpositive": numeric.notna() & np.isfinite(numeric)
            & numeric.le(0) & ~numeric.isin(IV_SENTINELS),
    }
    invalid = pd.Series(False, index=selected.index)
    for mask in reason_masks.values():
        invalid |= mask.fillna(False)
    selected.loc[invalid, "impl_volatility"] = np.nan
    selected["leg"] = selected["cp_flag"].map({"P": "iv_put_25d", "C": "iv_call_25d"})
    if selected.duplicated(["date", "leg"]).any():
        raise ValueError("Multiple 30-day 25-delta observations exist for a date/leg")
    daily = selected.pivot(index="date", columns="leg", values="impl_volatility").reset_index()
    daily.columns.name = None
    pair_valid = (
        np.isfinite(daily["iv_put_25d"]) & np.isfinite(daily["iv_call_25d"])
        & daily["iv_put_25d"].gt(0) & daily["iv_call_25d"].gt(0)
    )
    daily["valid_iv_pair"] = pair_valid
    daily["skew25_diff"] = np.where(
        pair_valid, daily["iv_put_25d"] - daily["iv_call_25d"], np.nan
    )
    daily["skew25_diff_volpts"] = 100.0 * daily["skew25_diff"]
    daily["skew25_ratio"] = np.where(
        pair_valid & daily["iv_call_25d"].gt(0),
        daily["iv_put_25d"] / daily["iv_call_25d"], np.nan,
    )
    daily = daily[[
        "date", "iv_put_25d", "iv_call_25d", "valid_iv_pair",
        "skew25_diff", "skew25_diff_volpts", "skew25_ratio",
    ]].sort_values("date")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.validity_audit.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output, index=False)
    complete = daily[daily["valid_iv_pair"]].copy()
    validity_rows = [
        {"stage": "selected_leg", "reason": reason, "count": int(mask.sum()),
         "disposition": "converted to NaN"}
        for reason, mask in reason_masks.items()
    ]
    validity_rows += [
        {"stage": "selected_leg", "reason": "invalid_any_reason", "count": int(invalid.sum()), "disposition": "converted to NaN"},
        {"stage": "daily_pair", "reason": "missing_or_invalid_put", "count": int((~np.isfinite(daily["iv_put_25d"]) | daily["iv_put_25d"].le(0)).sum()), "disposition": "pair metrics set to NaN"},
        {"stage": "daily_pair", "reason": "missing_or_invalid_call", "count": int((~np.isfinite(daily["iv_call_25d"]) | daily["iv_call_25d"].le(0)).sum()), "disposition": "pair metrics set to NaN"},
        {"stage": "daily_pair", "reason": "incomplete_or_invalid_pair", "count": int((~daily["valid_iv_pair"]).sum()), "disposition": "pair metrics set to NaN"},
        {"stage": "daily_pair", "reason": "valid_pair_retained", "count": int(daily["valid_iv_pair"].sum()), "disposition": "retained"},
    ]
    pd.DataFrame(validity_rows).to_csv(args.validity_audit, index=False)
    audit = {
        "input": str(args.input.resolve()),
        "raw_rows": len(raw),
        "raw_columns": raw_columns,
        "raw_date_min": str(raw["date"].min().date()),
        "raw_date_max": str(raw["date"].max().date()),
        "secid_counts": {str(k): int(v) for k, v in raw["secid"].value_counts().items()},
        "days_counts": {str(k): int(v) for k, v in raw["days"].value_counts().items()},
        "key_duplicates": key_duplicates,
        "missing_by_raw_field": {
            key: int(value) for key, value in raw[raw_columns].isna().sum().items()
        },
        "selected_rows": len(selected),
        "daily_rows": len(daily),
        "complete_pair_rows": len(complete),
        "incomplete_pair_rows": len(daily) - len(complete),
        "complete_date_min": str(complete["date"].min().date()),
        "complete_date_max": str(complete["date"].max().date()),
        "iv_failure_sentinels": sorted(IV_SENTINELS),
        "invalid_selected_rows_by_reason": {
            reason: int(mask.sum()) for reason, mask in reason_masks.items()
        },
        "invalid_selected_rows_any_reason": int(invalid.sum()),
        "validity_audit": str(args.validity_audit.resolve()),
        "summary": complete[["iv_put_25d", "iv_call_25d", "skew25_diff", "skew25_diff_volpts", "skew25_ratio"]].describe().to_dict(),
        "definition": "skew25_diff = IV_put(delta=-25, days=30) - IV_call(delta=+25, days=30)",
        "reporting_rescale": "skew25_diff_volpts = 100 * skew25_diff",
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(daily):,} daily rows ({len(complete):,} complete pairs) to {args.output}")


if __name__ == "__main__":
    main()
