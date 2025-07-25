"""
screen_filter.py
----------------

This module defines the ethical screening function for the Equity Factor Alpha
project. The function `moral_screen` removes equities that breach simple
AAOIFI-inspired thresholds on debt, cash, and non-compliant income.

See Section 2.2 of the project blueprint for details.
"""

import pandas as pd


def moral_screen(df_fund: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out stocks that do not meet ethical screening ratios.

    Criteria:
    - debt_to_equity < 0.33
    - cash_to_equity < 0.33
    - non_compliant_income_pct < 0.05
    """
    mask = (
        (df_fund["debt_to_equity"] < 0.33)
        & (df_fund["cash_to_equity"] < 0.33)
        & (df_fund["non_compliant_income_pct"] < 0.05)
    )
    return df_fund.loc[mask]
