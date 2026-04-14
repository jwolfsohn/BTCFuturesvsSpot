"""
Visualization of rolling price discovery dynamics matching the paper's Figure 3.

Three-panel plot (vertically stacked, shared x-axis):
  Panel A: BTC spot vs perp raw prices
  Panel B: Rolling d^P_{0,i} (price discovery betas) with reference lines at 0 and 1
  Panel C: Leadership shares — IPES vs NLS/PILS vs CovIS for spot market, with 50% ref

Reference: Shen, Huang, Zuo, Zivot (2026), Section 5.2, Figure 3.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_figure3(
    spot_prices: pd.Series,
    perp_prices: pd.Series,
    signals_df: pd.DataFrame,
    output_dir: str | Path = "results",
    title_suffix: str = "",
) -> Path:
    """
    Generate a three-panel figure replicating Figure 3 from the paper.

    Parameters
    ----------
    spot_prices  : raw BTC spot close prices (full resolution, for Panel A)
    perp_prices  : raw BTC perp close prices (full resolution, for Panel A)
    signals_df   : DataFrame from RollingResult.to_dataframe() — must have
                   d_P_0, d_P_1, ipes_0, nls_0, covis_0 columns
    output_dir   : directory for output files
    title_suffix : appended to the figure title

    Returns
    -------
    Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.08},
    )

    # ── Panel A: Raw prices ──────────────────────────────────────────
    ax1.plot(spot_prices.index, spot_prices.values,
             label="BTC Spot", color="#1f77b4", linewidth=0.7)
    ax1.plot(perp_prices.index, perp_prices.values,
             label="BTC Perp", color="#ff7f0e", linewidth=0.7)
    ax1.set_ylabel("Price (USDT)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"BTC Spot vs Perpetual Futures — Rolling Price Discovery{title_suffix}",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Panel B: Price Discovery Betas d^P_{0,i} ─────────────────────
    ts = signals_df.index
    ax2.plot(ts, signals_df["d_P_0"], label="d$^P$ Spot", color="#1f77b4", linewidth=1.0)
    ax2.plot(ts, signals_df["d_P_1"], label="d$^P$ Perp", color="#ff7f0e", linewidth=1.0)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Efficient (d$^P$=1)")
    ax2.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("d$^P_{0,i}$")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Shade conflict zones
    d_sum = signals_df["d_P_0"] + signals_df["d_P_1"]
    overshoot = d_sum > 2
    perverse = d_sum < 0
    if overshoot.any():
        ax2.fill_between(ts, ax2.get_ylim()[0], ax2.get_ylim()[1],
                         where=overshoot, alpha=0.08, color="red", label="Overshooting")
    if perverse.any():
        ax2.fill_between(ts, ax2.get_ylim()[0], ax2.get_ylim()[1],
                         where=perverse, alpha=0.08, color="blue", label="Perverse")

    # ── Panel C: Leadership shares (spot market = index 0) ────────────
    ax3.plot(ts, signals_df["ipes_0"], label="IPES (spot)",
             color="#2ca02c", linewidth=1.2)
    ax3.plot(ts, signals_df["nls_0"], label="NLS/PILS (spot)",
             color="#d62728", linewidth=1.0, linestyle="--")
    ax3.plot(ts, signals_df["covis_0"], label="CovIS (spot)",
             color="#9467bd", linewidth=1.0, linestyle=":")
    ax3.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                label="50% threshold")
    ax3.set_ylabel("Spot Leadership Share")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Time (UTC)")

    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    # Save
    png_path = output_dir / "figure3_btc_cascade.png"
    pdf_path = output_dir / "figure3_btc_cascade.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figure 3 saved: {png_path}")
    print(f"  Figure 3 saved: {pdf_path}")
    return png_path
