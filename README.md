# SKEW− and S&P 500 downside-risk forecasting

This repository contains the final code, result tables and figures for an MSc finance dissertation examining whether put-side option-implied skewness (SKEW−) adds useful information to VIX when forecasting S&P 500 downside risk.

## Research scope

The analysis covers:

- B&T-style monthly downturn logits at 1-, 3-, 6- and 12-month horizons;
- incremental comparisons against VIX;
- 21-trading-day downside-event, loss and semivariance outcomes;
- SKEW− level, monthly change, rolling z-score and state representations;
- 25-delta put–call IV differential and ratio comparisons; and
- recursive expanding-window forecasts for 2006–2017 and the 2018–2025 post-paper extension.

The final results use conventional model-specific inference. The 2018–2025 period is described as a post-paper extension, not a pristine holdout.

## Main result

The one-month B&T coefficient has the published negative direction but is not statistically significant. The six-month result is strongest. SKEW− levels add little information beyond VIX at the main one-month horizon, while monthly changes in SKEW− show the clearest association with DD21 loss and downside semivariance. Recursive forecast improvements are small and descriptive.

## Data sources

The underlying data are not included because they are licensed and too large for GitHub.

- **S&P 500 options and zero-coupon rates:** OptionMetrics IvyDB US, accessed through WRDS.
- **S&P 500 Index, VIX and published Cboe SKEW:** Bloomberg Terminal / Cboe.
- **Sample:** January 1996 to August 2025.

Place locally obtained source files under `data/` before running construction code. See [data/README.md](data/README.md) for the expected inputs.

## Repository contents

- `scripts/build_bt_strict_2_90.py`: BKM and put-side SKEW− construction.
- `scripts/build_25d_skew.py`: 25-delta differential and ratio construction.
- `scripts/prepare_tailrisk_specification_audit.py`: DD21, downside-semivariance and predictor preparation with leakage checks.
- `scripts/run_tailrisk_specification_audit.py`: in-sample and recursive OOS models.
- `scripts/create_thesis_figures.py`: produces the coefficient and OOS figures from the included final tables.
- `thesis/tables/`: dissertation-facing numerical results.
- `thesis/figures/`: the three final dissertation figures in PNG and PDF formats.

## Recreate the table-based figures

```powershell
python -m pip install -r requirements.txt
python scripts/create_thesis_figures.py
```

Figure 1 is supplied as a final rendered figure because its licensed monthly source series are not distributed. Figures 2 and 3 are regenerated from the included dissertation tables.

## Data-use note

Users must obtain OptionMetrics and Bloomberg/Cboe data through their own authorised subscriptions and comply with the applicable licence terms.
