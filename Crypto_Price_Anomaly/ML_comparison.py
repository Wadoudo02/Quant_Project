"""
ML_comparison.py
----------------

Utility script to compare the performance of the baseline Z‑score strategy
against the machine‑learning enhanced variant.  It expects that both
`run_mvp.py --use-ML False` and `run_mvp.py --use-ML True` have been executed
so that ``outputs/equity_curve.csv`` and ``outputs/equity_curve_ML.csv`` exist.

The script loads both equity curves, computes their Sharpe ratios and asserts
that the machine‑learning variant is **at least** as good as the baseline.  It
also overlays the two equity curves on a single plot for a visual comparison.
Use ``--show-plot`` to display this figure and ``--save-plot`` to write it to
``outputs/equity_comparison.pdf``.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import sharpe_ratio
from src.plotting import plot_equity


def main(args: argparse.Namespace) -> None:
    out_dir = Path("outputs")
    eq_base = (
        pd.read_csv(
            out_dir / "equity_curve.csv",
            index_col=0,
            parse_dates=True,
        )
        .iloc[:, 0]
    )
    eq_ml = (
        pd.read_csv(
            out_dir / "equity_curve_ML.csv",
            index_col=0,
            parse_dates=True,
        )
        .iloc[:, 0]
    )

    # Combined equity plot: baseline vs ML on the same axes
    ax = plot_equity(eq_base, title="Equity Curve – Baseline vs ML", show=False)
    ax.lines[0].set_label("Baseline")
    ax.plot(eq_ml.index, eq_ml.values, linewidth=1.4, label="ML")
    ax.legend()

    comparison_path = out_dir / "equity_comparison.pdf"
    if args.save_plot:
        ax.figure.savefig(comparison_path, dpi=300)
        print(f"Saved comparison plot to {comparison_path}")
    if args.show_plot:
        plt.show()

    ret_base = eq_base.diff().fillna(0)
    ret_ml = eq_ml.diff().fillna(0)
    sharpe_base = sharpe_ratio(ret_base)
    sharpe_ml = sharpe_ratio(ret_ml)

    print(f"Baseline Sharpe: {sharpe_base:.3f}")
    print(f"ML Sharpe      : {sharpe_ml:.3f}")
    if sharpe_ml < sharpe_base:
        raise AssertionError("ML Sharpe ratio is lower than baseline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline and ML equity curves and Sharpe ratios."
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the comparison plot using matplotlib.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save the comparison plot to outputs/equity_comparison.pdf.",
    )
    main(parser.parse_args())
