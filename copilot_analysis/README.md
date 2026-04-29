# Copilot analysis structure

This directory contains a cleaner notebook split for the fundkonto vs ISK decision and the follow-on fund-selection work.

## Notebook order

1. `01_isk_vs_fondkonto_baseline.ipynb`
   - Single-path baseline comparison.
   - Uses current assumptions and a long-horizon default.
   - Answers: what happens if the entire fondkonto is liquidated now and repurchased immediately inside the existing ISK?

2. `02_isk_break_even_sensitivity.ipynb`
   - Parameter sweeps across return and horizon.
   - Answers: when does ISK become better or worse than fondkonto under broad but realistic assumptions?

3. `03_fund_comparison_backtests.ipynb`
   - Backward-looking comparison of candidate funds.
   - Evaluates terminal value, volatility, and max drawdown.

4. `04_switch_timing_and_repurchase.ipynb`
   - Historical what-if timing study for selling Nordea Alfa and immediately buying another fund.
   - Answers: how much did switch timing matter historically?

5. `05_rules_based_fund_selection.ipynb`
   - Forward-looking, rules-based decision notebook.
   - Combines editable expected return assumptions with historical risk metrics.

## Shared module

`fund_analysis.py` centralizes CSV loading, Swedish number parsing, fund summary metrics, and baseline account-comparison helpers so the notebooks stay focused on analysis rather than plumbing.

## Notes

- All calculations currently assume the entire fondkonto can be treated as `nordea_alfa`.
- Taxes are parameterized, but defaults are prefilled for a 2026 baseline workflow.
- The baseline ISK model is intentionally simple and should be refined later if you want quarter-by-quarter tax-base modeling.
