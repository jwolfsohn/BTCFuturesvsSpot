#!/usr/bin/env python3
"""
Full pipeline: load 1s data -> estimate VECM/IPES -> backtest all portfolios.

Exploits the "Misguided Price Discovery" paper's finding that futures
overshoot during liquidation cascades. IPES detects this in real-time,
enabling profitable mean-reversion strategies.

Uses 1-second BTC/USDT spot and perpetual futures from Binance.

Usage:
    python3 -m scripts.run_pipeline
    python3 -m scripts.run_pipeline --skip-robustness

Outputs:
    results/resultsN.md -- Full results in Markdown (auto-increments)
    results/figure3_btc_cascade.png/pdf -- Three-panel visualization
"""
from __future__ import annotations

import argparse
import re
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress numpy RuntimeWarnings from VECM numerical edge cases
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scripts.load_data import (
    load_aligned_pair, load_funding_rate, load_spot, load_perp, validate,
)
from scripts.plot_paper_results import plot_figure3
from scripts.backtest import (
    robustness_sweep, run_full_pipeline,
    run_bootstrap_tests, lowz_threshold_sweep,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def _next_results_file() -> Path:
    """Return the next available results/resultsN.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for f in RESULTS_DIR.glob("results*.md")
        if (m := re.match(r"results(\d+)\.md$", f.name))
    ]
    n = max(existing, default=0) + 1
    return RESULTS_DIR / f"results{n}.md"


SWEEP_CSV = RESULTS_DIR / "robustness_sweep.csv"


# ── Helpers ─────────────────────────────────────────────────────────────────

def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"

def fmt_ratio(v: float) -> str:
    return f"{v:.3f}"

def metrics_to_md(df: pd.DataFrame) -> str:
    """Format a metrics DataFrame as a Markdown table."""
    display = df.copy()
    for col in ["total_return", "ann_return", "max_drawdown"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    for col in ["sharpe", "calmar"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_ratio)
    for col in ["avg_trade_pnl"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.6f}")
    for col in ["win_rate"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
    return display.to_markdown()


def section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


STRATEGY_OVERVIEW = """\
## Strategy Overview

Based on **"Misguided Price Discovery"** (Shen, Huang, Zuo, Zivot, 2026):
when BTC perpetual futures overshoot during liquidation cascades (d^P > 1),
IPES detects this in real-time while magnitude-based measures (NLS, CovIS)
incorrectly reward the overshooting. We exploit the predictable reversion.

| Portfolio | Strategy | Key Idea |
|-----------|----------|----------|
| P0 | Never Trade | Null benchmark |
| P1A/B | Distance Method | Raw spread z-scores (benchmark) |
| P2A/B | Cointegration | Cointegrating residual z-scores (benchmark) |
| P3 | IPES Filter | Full IPES entry gates: d^P>1, E_i threshold, shock z-scores |
| P4 | Coint + Stop-Loss | P2 with percentage stop-loss |
| **P5** | **Overshooting Fade** | **Fade when d^P > threshold in any market** |
| **P6** | **Conflict Zone Fade** | **Trade when d^P_0 + d^P_1 > threshold (overshooting zone)** |
| **P7** | **Transitory Scalper** | **Quick in/out on transitory shocks, tight exits** |
| **P8** | **IPES Adaptive** | **P3 with dynamic z_entry based on pricing error regime** |
| **P9** | **Cascade Fade** | **Target perp overshooting + high pricing error (cascade pattern)** |

"""


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run 1s VECM/IPES pipeline for BTC spot vs perp")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip the robustness sweep (saves significant time)")
    args = parser.parse_args()

    RESULTS_FILE = _next_results_file()
    out = StringIO()

    out.write("# Results — BTC Futures vs Spot Price Discovery Trading (1-Second)\n\n")
    out.write(f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}\n\n")
    out.write("Exploiting overshooting detection from the Misguided Price Discovery paper.\n")
    out.write("1-second BTC/USDT spot and perpetual futures from Binance.\n\n")

    out.write(STRATEGY_OVERVIEW)
    out.write("---\n")

    INTERVAL = "1s"

    # ── 1. Load and validate data ───────────────────────────────────────
    out.write(section("Data Summary"))

    print("Loading 1s BTC spot klines...")
    try:
        spot_c, perp_c = load_aligned_pair(symbol="BTCUSDT", interval="1s")
    except FileNotFoundError as e:
        print(f"\nERROR: 1s data not found. Run first:")
        print(f"  python3 -m scripts.download_data --paper")
        print(f"\nDetails: {e}")
        return

    print(f"  Loaded {len(spot_c):,} aligned 1s observations")
    print(f"  Range: {spot_c.index[0]} -> {spot_c.index[-1]}")

    # Validate
    spot_df = load_spot(interval="1s")
    perp_df = load_perp(interval="1s")
    spot_report = validate(spot_df, "BTC_SPOT_1s", "1s", quiet=True)
    perp_report = validate(perp_df, "BTC_PERP_1s", "1s", quiet=True)

    out.write(f"| Dataset | Rows | Range | Gaps |\n")
    out.write(f"|---------|------|-------|------|\n")
    out.write(f"| BTC Spot 1s | {spot_report['rows']:,} | "
              f"{spot_report['range_start'].strftime('%Y-%m-%d')} to "
              f"{spot_report['range_end'].strftime('%Y-%m-%d')} | "
              f"{len(spot_report['gaps'])} |\n")
    out.write(f"| BTC Perp 1s | {perp_report['rows']:,} | "
              f"{perp_report['range_start'].strftime('%Y-%m-%d')} to "
              f"{perp_report['range_end'].strftime('%Y-%m-%d')} | "
              f"{len(perp_report['gaps'])} |\n")
    out.write(f"| Aligned | {len(spot_c):,} | "
              f"{spot_c.index[0].strftime('%Y-%m-%d')} to "
              f"{spot_c.index[-1].strftime('%Y-%m-%d')} | -- |\n")

    # Funding rate context
    try:
        funding = load_funding_rate()
        fr_win = funding.loc["2024-01-01":"2024-01-05"]
        if len(fr_win) > 0:
            out.write(f"| Funding Rate | {len(fr_win)} obs | "
                      f"mean={fr_win.mean():.6f} | std={fr_win.std():.6f} |\n")
    except FileNotFoundError:
        pass
    out.write("\n")

    # Basis summary
    basis = spot_c - perp_c
    out.write(f"**Basis (spot - perp):** "
              f"mean={basis.mean():+.4f}, std={basis.std():.4f}, "
              f"min={basis.min():+.4f}, max={basis.max():+.4f}\n\n")

    # ── 2. Define analysis windows ──────────────────────────────────────
    windows = [
        {"name": "Full Period (Jan 1-5, 2024)",
         "start": "2024-01-01", "end": "2024-01-05 23:59:59"},
        {"name": "Cascade Day (Jan 3, 2024)",
         "start": "2024-01-03", "end": "2024-01-03 23:59:59"},
        {"name": "Pre-Cascade (Jan 1-2)",
         "start": "2024-01-01", "end": "2024-01-02 23:59:59"},
        {"name": "Post-Cascade (Jan 4-5)",
         "start": "2024-01-04", "end": "2024-01-05 23:59:59"},
    ]

    primary_params = dict(
        interval=INTERVAL,
        W=3600,               # 60-min window
        K=5,                  # VAR(5) -> VECM(4)
        deterministic="ci",   # Case 2 (constant inside CI equation)
        constrain_beta=True,  # beta=(1,-1)' enforced by funding rate mechanism
        E_threshold=0.5,
        tx_cost_bps=5,        # Conservative maker rates
    )

    out.write(f"**Primary parameters:** W={primary_params['W']} (60-min window), "
              f"step=300 (5-min), K={primary_params['K']}, "
              f"deterministic='{primary_params['deterministic']}', "
              f"beta_constrained={primary_params['constrain_beta']}, "
              f"E_threshold={primary_params['E_threshold']}, "
              f"tx_cost={primary_params['tx_cost_bps']} bps\n\n")

    # ── 3. Run full pipeline on each window ─────────────────────────────
    out.write(section("Portfolio Results by Period"))

    window_signals = {}
    window_prices = {}
    window_rolling = {}

    for w in windows:
        window_name = w["name"]
        out.write(section(window_name, level=3))

        # Slice to window
        spot_win = spot_c.loc[w["start"]:w["end"]]
        perp_win = perp_c.loc[w["start"]:w["end"]]

        # Re-align
        win_both = pd.concat([spot_win, perp_win], axis=1, join="inner").dropna()
        if len(win_both) < primary_params["W"] + 50:
            out.write(f"*Skipped — only {len(win_both)} observations "
                      f"(need >{primary_params['W'] + 50})*\n\n")
            continue

        c1 = win_both.iloc[:, 0]
        c2 = win_both.iloc[:, 1]

        print(f"\n{'='*60}")
        print(f"  {window_name}: {len(c1):,} obs")
        print(f"{'='*60}")

        t0 = time.time()
        rolling, portfolios, metrics = run_full_pipeline(
            c1, c2, **primary_params, verbose=True,
        )
        elapsed = time.time() - t0

        out.write(f"Observations: {len(c1):,} | "
                  f"Signals: {len(rolling.signals)} | "
                  f"Time: {elapsed:.1f}s\n\n")

        if len(metrics) > 0:
            out.write(metrics_to_md(metrics) + "\n\n")

            # Trade detail summary for key portfolios
            for p in portfolios:
                if p.n_trades > 0 and p.name.startswith((
                    "P1A", "P2A", "P3_Full", "P3_LowZ",
                    "P5_", "P6_", "P7_", "P8_", "P9_"
                )):
                    wins = sum(1 for t in p.trades if t.pnl_net > 0)
                    losses = p.n_trades - wins
                    out.write(f"**{p.name}**: {p.n_trades} trades "
                              f"({wins}W / {losses}L)")
                    if p.trades:
                        out.write(f" | Best: {max(t.pnl_net for t in p.trades):.6f}"
                                  f" | Worst: {min(t.pnl_net for t in p.trades):.6f}")
                    out.write("\n")
            out.write("\n")

            # Trade log for new strategies with trades
            for p in portfolios:
                if p.n_trades > 0 and p.name.startswith(("P5_O", "P6_C", "P7_T", "P8_", "P9_C")):
                    out.write(f"**{p.name} Trade Log (first 10):**\n\n")
                    out.write(f"| Entry | Exit | PnL (net) | Entry Reason | Exit Reason |\n")
                    out.write(f"|-------|------|-----------|--------------|-------------|\n")
                    for t in p.trades[:10]:
                        out.write(f"| {t.entry_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.exit_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.pnl_net:+.6f} "
                                  f"| {t.entry_reason[:40]} "
                                  f"| {t.exit_reason[:35]} |\n")
                    out.write("\n")
        else:
            out.write("*No signals produced.*\n\n")
            continue

        window_signals[window_name] = rolling.signals
        window_prices[window_name] = (c1, c2)
        window_rolling[window_name] = rolling

    # ── 4. Price Discovery Analysis (cascade day) ───────────────────────
    cascade_key = "Cascade Day (Jan 3, 2024)"
    if cascade_key in window_rolling:
        out.write(section("Price Discovery Analysis — Jan 3 Cascade"))

        df = window_rolling[cascade_key].to_dataframe()
        out.write(f"Rolling windows: {len(df)}\n\n")

        # d^P summary
        out.write("### Rolling d^P Summary\n\n")
        out.write("| Statistic | d^P Spot | d^P Perp |\n")
        out.write("|-----------|----------|----------|\n")
        for stat, fn in [("Mean", "mean"), ("Std", "std"), ("Min", "min"),
                         ("Median", "median"), ("Max", "max")]:
            v0 = getattr(df["d_P_0"], fn)()
            v1 = getattr(df["d_P_1"], fn)()
            out.write(f"| {stat} | {v0:.4f} | {v1:.4f} |\n")
        out.write("\n")

        # IPES vs NLS/PILS vs CovIS
        out.write("### Leadership Share Comparison (Spot Market)\n\n")
        out.write("The paper's key finding: IPES correctly assigns spot >50% "
                  "leadership during overshooting,\nwhile NLS/PILS and CovIS "
                  "incorrectly reward the overshooting market.\n\n")
        out.write("| Measure | Mean | Min | Median | Max |\n")
        out.write("|---------|------|-----|--------|-----|\n")
        for name, col in [("IPES", "ipes_0"), ("NLS/PILS", "nls_0"), ("CovIS", "covis_0")]:
            out.write(f"| {name} | {df[col].mean():.4f} | {df[col].min():.4f} | "
                      f"{df[col].median():.4f} | {df[col].max():.4f} |\n")
        out.write("\n")

        out.write("**Fraction of windows where spot is leader (share > 50%):**\n\n")
        for name, col in [("IPES", "ipes_0"), ("NLS/PILS", "nls_0"), ("CovIS", "covis_0")]:
            frac = (df[col] > 0.5).mean()
            out.write(f"- {name}: {frac:.1%}\n")
        out.write("\n")

        # Conflict zones
        out.write("### Conflict Zone Analysis\n\n")
        d_sum = df["d_P_0"] + df["d_P_1"]
        reliable = (d_sum >= 0) & (d_sum <= 2)
        overshooting = d_sum > 2
        perverse = d_sum < 0
        out.write(f"- Reliable zone (0 ≤ d^P_0 + d^P_1 ≤ 2): {reliable.mean():.1%}\n")
        out.write(f"- Overshooting zone (d^P_0 + d^P_1 > 2): {overshooting.mean():.1%}\n")
        out.write(f"- Perverse zone (d^P_0 + d^P_1 < 0): {perverse.mean():.1%}\n\n")

        # Generate Figure 3
        print("\nGenerating Figure 3...")
        try:
            spot_day = spot_c.loc["2024-01-03":"2024-01-03 23:59:59"]
            perp_day = perp_c.loc["2024-01-03":"2024-01-03 23:59:59"]
            plot_figure3(spot_day, perp_day, df, output_dir=RESULTS_DIR,
                         title_suffix=" — Jan 3, 2024 Liquidation Cascade")
        except Exception as e:
            print(f"  Warning: Figure 3 generation failed: {e}")
            out.write(f"*Figure 3 generation failed: {e}*\n\n")

    # ── 5. Bootstrap confidence intervals ─────────────────────────────
    out.write(section("Bootstrap Confidence Intervals (N=1000)"))
    out.write("Block bootstrap on signal streams. Drawdown differences: "
              "negative = strategy A has smaller drawdown (A better).\n\n")

    if not window_signals:
        out.write("*No windows had signals for bootstrap testing.*\n\n")
    else:
        # Run bootstrap on cascade day and full period
        boot_windows = [k for k in window_signals
                        if "Cascade" in k or "Full" in k]

        window_bootstrap = {}
        for wname in boot_windows:
            sigs = window_signals[wname]
            print(f"  Bootstrap: {wname}...")
            t0 = time.time()
            try:
                window_bootstrap[wname] = run_bootstrap_tests(
                    sigs,
                    E_threshold=primary_params["E_threshold"],
                    tx_cost_bps=primary_params["tx_cost_bps"],
                    n_bootstrap=1000,
                )
            except Exception as e:
                window_bootstrap[wname] = e
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

        # Drawdown CI table
        out.write(section("Drawdown Reduction vs P2A", level=3))
        out.write("| Window | Comparison | Observed dDD | 95% CI | p-value |\n")
        out.write("|--------|------------|-------------|--------|---------|\n")

        for wname in window_bootstrap:
            boot = window_bootstrap[wname]
            if isinstance(boot, Exception):
                out.write(f"| {wname[:40]} | -- | ERROR: {boot} | -- | -- |\n")
                continue
            for dd_test in boot["drawdown_tests"]:
                comp = f"{dd_test['a_name']} vs {dd_test['b_name']}"
                obs = dd_test["observed_diff"]
                ci_lo = dd_test["ci_lower"]
                ci_hi = dd_test["ci_upper"]
                pval = dd_test["p_value"]
                out.write(f"| {wname[:40]} | {comp} | "
                          f"{obs*100:+.2f}% | "
                          f"[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%] | "
                          f"{pval:.4f} |\n")
        out.write("\n")

        # PnL CI table
        out.write(section("Average Trade PnL: Bootstrap 95% CI", level=3))
        out.write("| Window | Portfolio | Avg PnL | 95% CI | N Trades |\n")
        out.write("|--------|-----------|---------|--------|----------|\n")

        for wname in window_bootstrap:
            boot = window_bootstrap[wname]
            if isinstance(boot, Exception):
                continue
            for pnl_test in boot["pnl_tests"]:
                port = pnl_test["portfolio"]
                mean_pnl = pnl_test["observed_mean"]
                ci_lo = pnl_test["ci_lower"]
                ci_hi = pnl_test["ci_upper"]
                nt = pnl_test["n_trades"]
                if nt > 0:
                    ci_str = f"[{ci_lo:.6f}, {ci_hi:.6f}]" if not np.isnan(ci_lo) else "--"
                    out.write(f"| {wname[:40]} | {port} | "
                              f"{mean_pnl:.6f} | {ci_str} | {nt} |\n")
        out.write("\n")

    # ── 6. LowZ threshold sweep ──────────────────────────────────────
    out.write(section("LowZ Threshold Sweep (z_entry Exploration)"))
    out.write("Sweeps z_entry from 0.25 to 2.0 on cascade day.\n\n")

    cascade_sigs = window_signals.get(cascade_key)
    if cascade_sigs:
        print(f"  LowZ sweep: {cascade_key}...")
        try:
            sweep_df = lowz_threshold_sweep(
                cascade_sigs,
                E_threshold=primary_params["E_threshold"],
                tx_cost_bps=primary_params["tx_cost_bps"],
            )
            display = sweep_df.copy()
            for col in ["total_return", "ann_return", "max_drawdown"]:
                if col in display.columns:
                    display[col] = display[col].map(fmt_pct)
            for col in ["sharpe", "calmar"]:
                if col in display.columns:
                    display[col] = display[col].map(fmt_ratio)
            for col in ["win_rate"]:
                if col in display.columns:
                    display[col] = display[col].map(lambda x: f"{x:.1%}")
            for col in ["avg_trade_pnl"]:
                if col in display.columns:
                    display[col] = display[col].map(lambda x: f"{x:.6f}")
            out.write(display.to_markdown(index=False) + "\n\n")
        except Exception as e:
            out.write(f"*Error: {e}*\n\n")
    else:
        out.write("*No cascade signals available.*\n\n")

    # ── 7. Robustness sweep ──────────────────────────────────────────
    if args.skip_robustness:
        out.write(section("Robustness Sweep"))
        out.write("*Skipped (--skip-robustness flag).*\n\n")
        print("  Robustness sweep: SKIPPED (--skip-robustness)")
    else:
        out.write(section("Robustness Sweep"))
        out.write("Parameter grid: W={1800,3600,7200}, K={3,5,7}, det={ci,co}, "
                  "beta={constrained,free}, E_i={0.3,0.5,0.75,1.0,1.5}, "
                  "tx={0,2,5,10,15} bps\n\n")

        # Run on cascade day only (most interesting)
        if cascade_key in window_prices:
            c1, c2 = window_prices[cascade_key]
            out.write(section(f"Robustness: {cascade_key}", level=3))
            print(f"  Robustness sweep: {cascade_key}...")
            t0 = time.time()
            try:
                sweep_df = robustness_sweep(c1, c2, interval=INTERVAL)
                if len(sweep_df) > 0:
                    sweep_df["window"] = cascade_key
                    profitable = sweep_df[sweep_df["total_return"] > 0]
                    out.write(f"Configurations tested: {len(sweep_df)} | "
                              f"Profitable: {len(profitable)} "
                              f"({len(profitable)/len(sweep_df)*100:.1f}%)\n\n")
                    has_trades = sweep_df[sweep_df["n_trades"] > 0].copy()
                    if len(has_trades) > 0:
                        top = has_trades.nlargest(10, "sharpe")
                        display_cols = ["W", "K", "deterministic", "beta_constrained",
                                       "E_threshold", "tx_cost_bps", "n_trades",
                                       "total_return", "sharpe", "win_rate", "max_drawdown"]
                        cols = [c for c in display_cols if c in top.columns]
                        out.write(top[cols].to_markdown(index=False) + "\n\n")
                    else:
                        out.write("*No configurations produced trades.*\n\n")
                    # Save sweep CSV
                    sweep_df.to_csv(SWEEP_CSV, index=False)
                else:
                    out.write("*Sweep returned empty results.*\n\n")
            except Exception as e:
                out.write(f"*Error: {e}*\n\n")
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")
        else:
            out.write("*No cascade data available for robustness sweep.*\n\n")

    # ── 8. Write results ────────────────────────────────────────────────
    out.write("---\n\n")
    out.write("> Generated by `python3 -m scripts.run_pipeline`\n")

    RESULTS_FILE.write_text(out.getvalue())
    print(f"\nResults written to {RESULTS_FILE}")
    if SWEEP_CSV.exists():
        print(f"Robustness sweep saved to {SWEEP_CSV}")


if __name__ == "__main__":
    main()
