from __future__ import annotations
"""Utility functions for fund comparisons and fondkonto-vs-ISK scenario analysis."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRADING_DAYS_PER_YEAR = 252

DEFAULT_PORTFOLIO = {
	"market_value": 773_517.0,
	"cost_basis": 181_574.76,
	"existing_isk_value": 630_000.0,
}


@dataclass(frozen=True)
class TaxAssumptions:
	"""Tax parameters used by account comparison scenarios."""

	capital_gains_tax_rate: float = 0.30
	isk_tax_rate: float = 0.01065
	isk_tax_free_amount: float = 300_000.0


@dataclass(frozen=True)
class AccountComparisonInputs:
	"""Inputs for comparing a liquidate-and-move ISK scenario against fondkonto."""

	market_value: float
	cost_basis: float
	existing_isk_value: float
	annual_return: float
	years: float
	tax: TaxAssumptions = TaxAssumptions()


def _parse_swedish_number(series: pd.Series) -> pd.Series:
	"""Parse Swedish-formatted numeric text into numeric values.

	Handles thousands separator dot and decimal comma.
	"""

	return (
		series.astype(str)
		.str.replace('"', "", regex=False)
		.str.replace(" ", "", regex=False)
		.str.replace(".", "", regex=False)
		.str.replace(",", ".", regex=False)
		.replace({"": np.nan})
		.pipe(pd.to_numeric, errors="coerce")
	)


def _parse_swedish_percent(series: pd.Series) -> pd.Series:
	"""Parse percentage strings like 1,23% into decimal values like 0.0123."""

	cleaned = series.astype(str).str.replace("%", "", regex=False)
	return _parse_swedish_number(cleaned) / 100.0


def load_fund_nav(csv_path: str | Path) -> pd.DataFrame:
	"""Load one fund CSV and normalize date, NAV, and daily return columns."""

	csv_path = Path(csv_path)
	frame = pd.read_csv(csv_path)
	frame = frame.rename(columns={"Datum": "date", "Senaste": "nav", "+/- %": "daily_return"})
	frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
	frame["nav"] = _parse_swedish_number(frame["nav"])
	frame["daily_return"] = _parse_swedish_percent(frame["daily_return"])
	frame = frame[["date", "nav", "daily_return"]].sort_values("date").dropna(subset=["nav"])
	frame = frame.drop_duplicates(subset=["date"]).reset_index(drop=True)
	frame["fund"] = csv_path.stem
	return frame


def load_all_funds(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
	"""Load and concatenate all fund CSV files from a directory."""

	data_dir = Path(data_dir)
	frames = [load_fund_nav(path) for path in sorted(data_dir.glob("*.csv"))]
	if not frames:
		raise FileNotFoundError(f"No CSV files found in {data_dir}")
	return pd.concat(frames, ignore_index=True)


def add_return_index(frame: pd.DataFrame, value_column: str = "nav") -> pd.DataFrame:
	"""Add daily percentage change and normalized growth index to a fund frame."""

	ordered = frame.sort_values("date").copy()
	ordered["daily_pct_change"] = ordered[value_column].pct_change()
	first_value = ordered[value_column].iloc[0]
	ordered["growth_of_1kr"] = ordered[value_column] / first_value
	return ordered


def annualized_return(nav: pd.Series, dates: pd.Series | None = None) -> float:
	"""Compute annualized return.

	If dates are provided, annualization is based on elapsed calendar time.
	Otherwise, observations are converted with an assumed trading-day year.
	"""

	nav = nav.dropna()
	if len(nav) < 2:
		return np.nan
	total_return = nav.iloc[-1] / nav.iloc[0]
	if dates is not None:
		ordered_dates = pd.to_datetime(dates).sort_values()
		elapsed_days = (ordered_dates.iloc[-1] - ordered_dates.iloc[0]).days
		years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
	else:
		years = len(nav) / TRADING_DAYS_PER_YEAR
	if pd.isna(years) or years <= 0:
		return np.nan
	return total_return ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
	"""Compute annualized volatility from daily returns."""

	daily_returns = daily_returns.dropna()
	if len(daily_returns) < 2:
		return np.nan
	return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(nav: pd.Series) -> float:
	"""Return the worst peak-to-trough drawdown in the NAV series."""

	nav = nav.dropna()
	if nav.empty:
		return np.nan
	running_max = nav.cummax()
	drawdown = nav / running_max - 1
	return drawdown.min()


def summarize_fund(frame: pd.DataFrame) -> dict[str, float | str | pd.Timestamp]:
	"""Build summary metrics for one fund time series."""

	ordered = add_return_index(frame)
	return {
		"fund": ordered["fund"].iloc[0],
		"start_date": ordered["date"].iloc[0],
		"end_date": ordered["date"].iloc[-1],
		"observations": len(ordered),
		"cagr": annualized_return(ordered["nav"], ordered["date"]),
		"volatility": annualized_volatility(ordered["daily_pct_change"]),
		"max_drawdown": max_drawdown(ordered["nav"]),
		"ending_multiple": ordered["growth_of_1kr"].iloc[-1],
	}


def fund_summary_table(all_funds: pd.DataFrame) -> pd.DataFrame:
	"""Create a summary table with one row per fund."""

	summaries = [summarize_fund(group) for _, group in all_funds.groupby("fund")]
	return pd.DataFrame(summaries).sort_values("cagr", ascending=False).reset_index(drop=True)


def future_value(principal: float, annual_return: float, years: float) -> float:
	"""Compound principal at a constant annual return for a number of years."""

	return principal * (1 + annual_return) ** years


def realized_gain(market_value: float, cost_basis: float) -> float:
	"""Return taxable realized gain, never below zero."""

	return max(market_value - cost_basis, 0.0)


def liquidation_tax(market_value: float, cost_basis: float, tax_rate: float) -> float:
	"""Compute tax due when liquidating holdings with a given gain tax rate."""

	return realized_gain(market_value, cost_basis) * tax_rate


def net_capital_after_liquidation(market_value: float, cost_basis: float, tax_rate: float) -> float:
	"""Return capital available for reinvestment after liquidation tax."""

	return market_value - liquidation_tax(market_value, cost_basis, tax_rate)


def compare_accounts(inputs: AccountComparisonInputs) -> dict[str, float]:
	"""Compare moved-capital outcomes for ISK and fondkonto.

	This uses a marginal comparison for the moved capital and excludes tax effects
	from already-existing ISK assets in the delta result, except for interaction
	through a shared ISK tax-free allowance.
	"""

	tax_now = liquidation_tax(
		market_value=inputs.market_value,
		cost_basis=inputs.cost_basis,
		tax_rate=inputs.tax.capital_gains_tax_rate,
	)
	investable_isk_capital = net_capital_after_liquidation(
		market_value=inputs.market_value,
		cost_basis=inputs.cost_basis,
		tax_rate=inputs.tax.capital_gains_tax_rate,
	)
	isk_future_value_before_annual_tax = future_value(
		principal=investable_isk_capital,
		annual_return=inputs.annual_return,
		years=inputs.years,
	)
	taxable_isk_base_total = max(
		inputs.existing_isk_value + investable_isk_capital - inputs.tax.isk_tax_free_amount,
		0.0,
	)
	taxable_isk_base_existing = max(inputs.existing_isk_value - inputs.tax.isk_tax_free_amount, 0.0)
	taxable_isk_base_marginal = max(taxable_isk_base_total - taxable_isk_base_existing, 0.0)
	annual_isk_tax = taxable_isk_base_marginal * inputs.tax.isk_tax_rate
	annual_isk_tax_total_account = taxable_isk_base_total * inputs.tax.isk_tax_rate
	isk_tax_drag_future_value = annual_isk_tax * inputs.years
	isk_terminal_value = isk_future_value_before_annual_tax - isk_tax_drag_future_value

	fondkonto_future_value = future_value(
		principal=inputs.market_value,
		annual_return=inputs.annual_return,
		years=inputs.years,
	)
	fondkonto_tax_at_exit = liquidation_tax(
		market_value=fondkonto_future_value,
		cost_basis=inputs.cost_basis,
		tax_rate=inputs.tax.capital_gains_tax_rate,
	)
	fondkonto_terminal_value = fondkonto_future_value - fondkonto_tax_at_exit

	return {
		"investable_isk_capital": investable_isk_capital,
		"tax_paid_now": tax_now,
		"isk_tax_free_amount": inputs.tax.isk_tax_free_amount,
		"taxable_isk_base_marginal": taxable_isk_base_marginal,
		"annual_isk_tax": annual_isk_tax,
		"taxable_isk_base_total_account": taxable_isk_base_total,
		"annual_isk_tax_total_account": annual_isk_tax_total_account,
		"isk_terminal_value": isk_terminal_value,
		"fondkonto_terminal_value": fondkonto_terminal_value,
		"difference_isk_minus_fondkonto": isk_terminal_value - fondkonto_terminal_value,
	}


def compare_accounts_grid(
	market_value: float,
	cost_basis: float,
	existing_isk_value: float,
	annual_returns: list[float] | np.ndarray,
	years: list[float] | np.ndarray,
	tax: TaxAssumptions | None = None,
) -> pd.DataFrame:
	"""Evaluate account comparison over a grid of return and horizon assumptions."""

	tax = tax or TaxAssumptions()
	rows = []
	for annual_return in annual_returns:
		for horizon in years:
			result = compare_accounts(
				AccountComparisonInputs(
					market_value=market_value,
					cost_basis=cost_basis,
					existing_isk_value=existing_isk_value,
					annual_return=float(annual_return),
					years=float(horizon),
					tax=tax,
				)
			)
			rows.append(
				{
					"annual_return": float(annual_return),
					"years": float(horizon),
					**result,
				}
			)
	return pd.DataFrame(rows)


def score_funds(summary_table: pd.DataFrame) -> pd.DataFrame:
	"""Score funds by return, volatility, and drawdown into a composite rank."""

	scored = summary_table.copy()
	scored["cagr_score"] = scored["cagr"].rank(pct=True)
	scored["volatility_score"] = 1 - scored["volatility"].rank(pct=True)
	scored["drawdown_score"] = scored["max_drawdown"].rank(pct=True)
	scored["composite_score"] = (
		0.5 * scored["cagr_score"]
		+ 0.25 * scored["volatility_score"]
		+ 0.25 * scored["drawdown_score"]
	)
	return scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
