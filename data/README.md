# Data inputs

Raw market and option data are intentionally excluded from version control.

The analysis used:

- OptionMetrics IvyDB US S&P 500 option quotes;
- the OptionMetrics zero curve for maturity-matched risk-free rates;
- Bloomberg/Cboe series for the S&P 500 Index, VIX and published Cboe SKEW; and
- an end-of-month sample from January 1996 through August 2025.

To reproduce the construction, place authorised local extracts under this directory using the paths expected by the scripts. Do not commit licensed source extracts to Git.
