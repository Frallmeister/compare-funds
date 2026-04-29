import pandas as pd

from fund_analysis import (
    AccountComparisonInputs,
    TaxAssumptions,
    annualized_return,
    compare_accounts,
    liquidation_tax,
    net_capital_after_liquidation,
    score_funds,
)


def test_existing_isk_balance_does_not_change_delta_without_allowance() -> None:
    tax = TaxAssumptions(capital_gains_tax_rate=0.30, isk_tax_rate=0.00888, isk_tax_free_amount=0.0)
    base = AccountComparisonInputs(
        market_value=773_517.0,
        cost_basis=181_574.76,
        existing_isk_value=0.0,
        annual_return=0.07,
        years=30,
        tax=tax,
    )
    with_existing = AccountComparisonInputs(
        market_value=773_517.0,
        cost_basis=181_574.76,
        existing_isk_value=630_000.0,
        annual_return=0.07,
        years=30,
        tax=tax,
    )

    base_result = compare_accounts(base)
    existing_result = compare_accounts(with_existing)

    assert round(base_result["difference_isk_minus_fondkonto"], 6) == round(
        existing_result["difference_isk_minus_fondkonto"], 6
    )


def test_existing_isk_uses_up_allowance_and_increases_marginal_isk_tax() -> None:
    base = AccountComparisonInputs(
        market_value=773_517.0,
        cost_basis=181_574.76,
        existing_isk_value=0.0,
        annual_return=0.07,
        years=30,
        tax=TaxAssumptions(isk_tax_free_amount=300_000.0),
    )
    with_existing = AccountComparisonInputs(
        market_value=773_517.0,
        cost_basis=181_574.76,
        existing_isk_value=630_000.0,
        annual_return=0.07,
        years=30,
        tax=TaxAssumptions(isk_tax_free_amount=300_000.0),
    )

    base_result = compare_accounts(base)
    existing_result = compare_accounts(with_existing)

    assert base_result["annual_isk_tax"] < existing_result["annual_isk_tax"]


def test_drawdown_scoring_prefers_smaller_drawdowns() -> None:
    summary = pd.DataFrame(
        {
            "fund": ["better_drawdown", "worse_drawdown"],
            "cagr": [0.08, 0.08],
            "volatility": [0.2, 0.2],
            "max_drawdown": [-0.2, -0.5],
            "ending_multiple": [2.0, 2.0],
        }
    )

    scored = score_funds(summary)

    assert scored.iloc[0]["fund"] == "better_drawdown"


def test_net_capital_after_liquidation_uses_realized_gain_tax() -> None:
    market_value = 773_517.0
    cost_basis = 181_574.76
    tax_rate = 0.30

    expected_tax = liquidation_tax(market_value, cost_basis, tax_rate)
    expected_capital = market_value - expected_tax

    assert round(net_capital_after_liquidation(market_value, cost_basis, tax_rate), 6) == round(
        expected_capital, 6
    )


def test_annualized_return_uses_calendar_dates_when_available() -> None:
    nav = pd.Series([100.0, 110.0])
    dates = pd.to_datetime(pd.Series(["2020-01-01", "2021-01-01"]))

    result = annualized_return(nav, dates)

    assert 0.09 < result < 0.11
