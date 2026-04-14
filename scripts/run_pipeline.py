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

import re
import time
import warnings
from io import StringIO
from pathlib import Path

import pandas as pd

# Suppress numpy RuntimeWarnings from VECM numerical edge cases
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scripts.load_data import (
    load_aligned_pair, load_funding_rate,
)
from scripts.plot_paper_results import plot_figure3
from scripts.backtest import (
    run_full_pipeline,
    backtest_p_mom, backtest_p_carry, backtest_p_dir,
    p_mom_sweep, p_carry_sweep, p_dir_sweep,
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
    for col in ["avg_trade_pnl", "avg_trade_pnl_gross"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.6f}")
    for col in ["win_rate", "gross_win_rate"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
    for col in ["breakeven_tc_bps"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.1f}")
    return display.to_markdown()


def section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


STRATEGY_OVERVIEW = """\
## Strategy Overview

**IPES as regime filter**: IPES measures real-time market microstructure quality.
Instead of using it as a trade signal, we use it to modulate risk across strategies.

| Portfolio | Type | Strategy | Key Idea |
|-----------|------|----------|----------|
| P0 | Benchmark | Never Trade | Null benchmark |
| P_BH | Benchmark | Buy & Hold BTC | Directional benchmark |
| P2A | Spread | Cointegration | Mean-reversion benchmark (z-score entry/exit) |
| P_AVOID | Spread | IPES Avoid-Stress | P2A but skip trades during IPES-detected stress |
| P_DIR | Directional | Cascade Fade | Directional bet on perp recovery after overshooting |
| **P_MOM** | **Always-on** | **Momentum + IPES** | **Trend-follow BTC; IPES throttles risk in stressed regimes** |
| **P_CARRY** | **Always-on** | **Carry + IPES** | **Earn funding rate; IPES shields against cascade losses** |

"""


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    RESULTS_FILE = _next_results_file()
    out = StringIO()

    out.write("# Results — BTC IPES Regime-Filtered Trading System\n\n")
    out.write(f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}\n\n")
    out.write("IPES as a real-time market quality filter for always-on BTC strategies.\n")
    out.write("1-minute full year 2024 + 1-second cascade deep-dives.\n\n")

    out.write(STRATEGY_OVERVIEW)
    out.write("---\n")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: FULL-YEAR BACKTEST (1-minute data, always-on strategies)
    # ══════════════════════════════════════════════════════════════════
    out.write(section("Part 1: Full-Year Backtest (1-Minute, 2024)"))

    print("=" * 60)
    print("  PART 1: Full-Year 1-Minute Backtest")
    print("=" * 60)

    print("Loading 1m BTC data...")
    try:
        spot_1m, perp_1m = load_aligned_pair(symbol="BTCUSDT", interval="1m")
    except FileNotFoundError as e:
        print(f"\nERROR: 1m data not found. Run: python3 -m scripts.download_data")
        print(f"\nDetails: {e}")
        return

    print(f"  Loaded {len(spot_1m):,} aligned 1m observations")
    print(f"  Range: {spot_1m.index[0]} -> {spot_1m.index[-1]}")

    # Data summary
    out.write(section("Data Summary", level=3))
    out.write(f"| Dataset | Rows | Range |\n")
    out.write(f"|---------|------|-------|\n")
    out.write(f"| Aligned 1m | {len(spot_1m):,} | "
              f"{spot_1m.index[0].strftime('%Y-%m-%d')} to "
              f"{spot_1m.index[-1].strftime('%Y-%m-%d')} |\n")

    # Funding rate for full year
    try:
        funding = load_funding_rate()
        out.write(f"| Funding Rate | {len(funding)} obs | "
                  f"mean={funding.mean():.6f} | std={funding.std():.6f} |\n")
    except FileNotFoundError:
        funding = None
    out.write("\n")

    basis_1m = spot_1m - perp_1m
    out.write(f"**Basis:** mean={basis_1m.mean():+.2f}, std={basis_1m.std():.2f}, "
              f"min={basis_1m.min():+.2f}, max={basis_1m.max():+.2f}\n")
    out.write(f"**BTC Price:** {spot_1m.iloc[0]:.0f} (Jan 1) → "
              f"{spot_1m.iloc[-1]:.0f} (Dec 31) "
              f"({(spot_1m.iloc[-1]/spot_1m.iloc[0]-1)*100:+.1f}%)\n\n")

    # Full-year analysis windows
    fy_windows = [
        {"name": "H1 2024 (Jan-Jun)",
         "start": "2024-01-01", "end": "2024-06-30 23:59:59",
         "role": "train"},
        {"name": "H2 2024 (Jul-Dec)",
         "start": "2024-07-01", "end": "2024-12-31 23:59:59",
         "role": "test"},
        {"name": "Full Year 2024",
         "start": "2024-01-01", "end": "2024-12-31 23:59:59",
         "role": "reference"},
    ]

    primary_params_1m = dict(
        interval="1m",
        W=360,                # 6-hour window (for 1m data)
        K=5,                  # VAR(5) -> VECM(4)
        deterministic="ci",
        constrain_beta=True,
        E_threshold=0.5,
        tx_cost_bps=5,
    )

    out.write(f"**Parameters:** W={primary_params_1m['W']} (6-hour window), "
              f"step=15 (15-min), K={primary_params_1m['K']}, "
              f"tx_cost={primary_params_1m['tx_cost_bps']} bps\n\n")

    out.write(section("Full-Year Portfolio Results", level=3))

    fy_signals = {}
    fy_rolling = {}

    for w in fy_windows:
        wname = w["name"]
        out.write(section(wname, level=4))

        spot_win = spot_1m.loc[w["start"]:w["end"]]
        perp_win = perp_1m.loc[w["start"]:w["end"]]
        win_both = pd.concat([spot_win, perp_win], axis=1, join="inner").dropna()

        if len(win_both) < primary_params_1m["W"] + 50:
            out.write(f"*Skipped — only {len(win_both)} observations*\n\n")
            continue

        c1 = win_both.iloc[:, 0]
        c2 = win_both.iloc[:, 1]

        print(f"\n  {wname}: {len(c1):,} obs")
        t0 = time.time()
        rolling, portfolios, metrics = run_full_pipeline(
            c1, c2, **primary_params_1m, verbose=False,
        )
        elapsed = time.time() - t0

        out.write(f"Observations: {len(c1):,} | "
                  f"Signals: {len(rolling.signals)} | "
                  f"Time: {elapsed:.1f}s\n\n")

        if len(metrics) > 0:
            out.write(metrics_to_md(metrics) + "\n\n")

            # Trade summaries for active strategies
            for p in portfolios:
                if p.n_trades > 0:
                    wins = sum(1 for t in p.trades if t.pnl_net > 0)
                    wins_gross = sum(1 for t in p.trades if t.pnl_gross > 0)
                    losses = p.n_trades - wins
                    out.write(f"**{p.name}**: {p.n_trades} trades "
                              f"({wins}W / {losses}L, "
                              f"{wins_gross} gross winners)")
                    if p.trades:
                        out.write(f" | Best: {max(t.pnl_net for t in p.trades):.6f}"
                                  f" | Worst: {min(t.pnl_net for t in p.trades):.6f}")
                    out.write("\n")
            out.write("\n")

            # Trade log for key strategies
            for p in portfolios:
                if 0 < p.n_trades <= 20 and p.name in ("P_MOM", "P_CARRY", "P_DIR"):
                    out.write(f"**{p.name} Trade Log (first 10):**\n\n")
                    out.write(f"| Entry | Exit | PnL (gross) | PnL (net) | Entry Reason | Exit Reason |\n")
                    out.write(f"|-------|------|-------------|-----------|--------------|-------------|\n")
                    for t in p.trades[:10]:
                        out.write(f"| {t.entry_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.exit_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.pnl_gross:+.6f} "
                                  f"| {t.pnl_net:+.6f} "
                                  f"| {t.entry_reason[:40]} "
                                  f"| {t.exit_reason[:40]} |\n")
                    out.write("\n")

        fy_signals[wname] = rolling.signals
        fy_rolling[wname] = rolling

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD VALIDATION: Train on H1, test on H2
    # ══════════════════════════════════════════════════════════════════
    h1_key = "H1 2024 (Jan-Jun)"
    h2_key = "H2 2024 (Jul-Dec)"

    if h1_key in fy_signals and h2_key in fy_signals:
        out.write(section("Walk-Forward Validation (Train H1 → Test H2)"))
        out.write("Parameters selected on H1, frozen, then tested on H2. "
                  "This is the honest out-of-sample test.\n\n")

        h1_sigs = fy_signals[h1_key]
        h2_sigs = fy_signals[h2_key]

        print("\n" + "=" * 60)
        print("  WALK-FORWARD VALIDATION")
        print("=" * 60)

        # ── P_MOM walk-forward ──────────────────────────────────────
        print("\n  P_MOM parameter sweep on H1...")
        h1_mom = p_mom_sweep(h1_sigs, tx_cost_bps=primary_params_1m["tx_cost_bps"],
                             verbose=True)

        # Select best: Sharpe > 0, then highest total_return
        viable_mom = h1_mom[h1_mom["sharpe"] > 0]
        if len(viable_mom) > 0:
            best_mom = viable_mom.loc[viable_mom["total_return"].idxmax()]
        else:
            best_mom = h1_mom.loc[h1_mom["total_return"].idxmax()]

        sel_mom = {
            "ema_periods": int(best_mom["ema_periods"]),
            "E_threshold_red": float(best_mom["E_threshold_red"]),
            "dp_red_threshold": float(best_mom["dp_red_threshold"]),
            "ema_band_pct": float(best_mom["ema_band_pct"]),
        }

        out.write(section("P_MOM Walk-Forward", level=3))
        out.write(f"**H1 sweep:** {len(h1_mom)} configs, "
                  f"{(h1_mom['sharpe'] > 0).sum()} with Sharpe > 0\n\n")
        out.write(f"**Selected params (from H1):** {sel_mom}\n")
        out.write(f"**H1 performance:** return={best_mom['total_return']:.2%}, "
                  f"sharpe={best_mom['sharpe']:.2f}, "
                  f"trades={int(best_mom['n_trades'])}\n\n")

        # Test on H2
        h2_mom_wf = backtest_p_mom(h2_sigs, **sel_mom,
                                    tx_cost_bps=primary_params_1m["tx_cost_bps"])
        h2_mom_default = backtest_p_mom(h2_sigs,
                                         tx_cost_bps=primary_params_1m["tx_cost_bps"])
        wf_m = h2_mom_wf.metrics()
        df_m = h2_mom_default.metrics()

        out.write("**H2 out-of-sample comparison:**\n\n")
        out.write("| Config | Return | Sharpe | Max DD | Trades | Breakeven TC |\n")
        out.write("|--------|--------|--------|--------|--------|--------------|\n")
        out.write(f"| H1-selected | {wf_m['total_return']:.2%} | "
                  f"{wf_m['sharpe']:.2f} | {wf_m['max_drawdown']:.2%} | "
                  f"{wf_m['n_trades']} | {wf_m['breakeven_tc_bps']:.1f} |\n")
        out.write(f"| Hardcoded defaults | {df_m['total_return']:.2%} | "
                  f"{df_m['sharpe']:.2f} | {df_m['max_drawdown']:.2%} | "
                  f"{df_m['n_trades']} | {df_m['breakeven_tc_bps']:.1f} |\n\n")

        # Parameter sensitivity by ema_periods
        out.write("**Parameter sensitivity (H1, grouped by ema_periods):**\n\n")
        out.write("| ema_periods | Mean Return | Std Return | Sharpe>0 frac | Mean Trades |\n")
        out.write("|-------------|-------------|------------|---------------|-------------|\n")
        for ep in sorted(h1_mom["ema_periods"].unique()):
            subset = h1_mom[h1_mom["ema_periods"] == ep]
            out.write(f"| {ep} | {subset['total_return'].mean():.2%} | "
                      f"{subset['total_return'].std():.2%} | "
                      f"{(subset['sharpe'] > 0).mean():.0%} | "
                      f"{subset['n_trades'].mean():.0f} |\n")
        out.write("\n")

        # ── P_CARRY walk-forward ────────────────────────────────────
        print("  P_CARRY parameter sweep on H1...")
        h1_carry = p_carry_sweep(h1_sigs,
                                  tx_cost_bps=primary_params_1m["tx_cost_bps"],
                                  verbose=True)

        viable_carry = h1_carry[h1_carry["sharpe"] > 0]
        if len(viable_carry) > 0:
            best_carry = viable_carry.loc[viable_carry["total_return"].idxmax()]
        else:
            best_carry = h1_carry.loc[h1_carry["total_return"].idxmax()]

        sel_carry = {
            "E_threshold": float(best_carry["E_threshold"]),
            "d_sum_threshold": float(best_carry["d_sum_threshold"]),
            "reentry_cooldown": int(best_carry["reentry_cooldown"]),
        }

        out.write(section("P_CARRY Walk-Forward", level=3))
        out.write(f"**H1 sweep:** {len(h1_carry)} configs, "
                  f"{(h1_carry['sharpe'] > 0).sum()} with Sharpe > 0\n\n")
        out.write(f"**Selected params (from H1):** {sel_carry}\n")
        out.write(f"**H1 performance:** return={best_carry['total_return']:.2%}, "
                  f"sharpe={best_carry['sharpe']:.2f}, "
                  f"trades={int(best_carry['n_trades'])}\n\n")

        h2_carry_wf = backtest_p_carry(h2_sigs, **sel_carry,
                                        tx_cost_bps=primary_params_1m["tx_cost_bps"])
        h2_carry_default = backtest_p_carry(h2_sigs,
                                             tx_cost_bps=primary_params_1m["tx_cost_bps"])
        wf_c = h2_carry_wf.metrics()
        df_c = h2_carry_default.metrics()

        out.write("**H2 out-of-sample comparison:**\n\n")
        out.write("| Config | Return | Sharpe | Max DD | Trades | Breakeven TC |\n")
        out.write("|--------|--------|--------|--------|--------|--------------|\n")
        out.write(f"| H1-selected | {wf_c['total_return']:.2%} | "
                  f"{wf_c['sharpe']:.2f} | {wf_c['max_drawdown']:.2%} | "
                  f"{wf_c['n_trades']} | {wf_c['breakeven_tc_bps']:.1f} |\n")
        out.write(f"| Hardcoded defaults | {df_c['total_return']:.2%} | "
                  f"{df_c['sharpe']:.2f} | {df_c['max_drawdown']:.2%} | "
                  f"{df_c['n_trades']} | {df_c['breakeven_tc_bps']:.1f} |\n\n")

        # ── P_DIR walk-forward ──────────────────────────────────────
        print("  P_DIR parameter sweep on H1...")
        h1_dir = p_dir_sweep(h1_sigs,
                              tx_cost_bps=primary_params_1m["tx_cost_bps"],
                              verbose=True)

        viable_dir = h1_dir[h1_dir["sharpe"] > 0]
        if len(viable_dir) > 0:
            best_dir = viable_dir.loc[viable_dir["total_return"].idxmax()]
        else:
            best_dir = h1_dir.loc[h1_dir["total_return"].idxmax()]

        sel_dir = {
            "dp_threshold": float(best_dir["dp_threshold"]),
            "stop_loss_pct": float(best_dir["stop_loss_pct"]),
            "trail_activation_pct": float(best_dir["trail_activation_pct"]),
            "max_hold_signals": int(best_dir["max_hold_signals"]),
        }

        out.write(section("P_DIR Walk-Forward", level=3))
        out.write(f"**H1 sweep:** {len(h1_dir)} configs, "
                  f"{(h1_dir['sharpe'] > 0).sum()} with Sharpe > 0\n\n")
        out.write(f"**Selected params (from H1):** {sel_dir}\n")
        out.write(f"**H1 performance:** return={best_dir['total_return']:.2%}, "
                  f"sharpe={best_dir['sharpe']:.2f}, "
                  f"trades={int(best_dir['n_trades'])}\n\n")

        h2_dir_wf = backtest_p_dir(h2_sigs, **sel_dir,
                                    tx_cost_bps=primary_params_1m["tx_cost_bps"])
        h2_dir_default = backtest_p_dir(h2_sigs,
                                         tx_cost_bps=primary_params_1m["tx_cost_bps"])
        wf_d = h2_dir_wf.metrics()
        df_d = h2_dir_default.metrics()

        out.write("**H2 out-of-sample comparison:**\n\n")
        out.write("| Config | Return | Sharpe | Max DD | Trades | Breakeven TC |\n")
        out.write("|--------|--------|--------|--------|--------|--------------|\n")
        out.write(f"| H1-selected | {wf_d['total_return']:.2%} | "
                  f"{wf_d['sharpe']:.2f} | {wf_d['max_drawdown']:.2%} | "
                  f"{wf_d['n_trades']} | {wf_d['breakeven_tc_bps']:.1f} |\n")
        out.write(f"| Hardcoded defaults | {df_d['total_return']:.2%} | "
                  f"{df_d['sharpe']:.2f} | {df_d['max_drawdown']:.2%} | "
                  f"{df_d['n_trades']} | {df_d['breakeven_tc_bps']:.1f} |\n\n")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: CASCADE DEEP-DIVES (1-second data)
    # ══════════════════════════════════════════════════════════════════
    out.write(section("Part 2: Cascade Deep-Dives (1-Second)"))

    print("\n" + "=" * 60)
    print("  PART 2: 1-Second Cascade Deep-Dives")
    print("=" * 60)

    print("Loading 1s BTC data...")
    try:
        spot_c, perp_c = load_aligned_pair(symbol="BTCUSDT", interval="1s")
    except FileNotFoundError as e:
        print(f"  1s data not available, skipping cascade deep-dives")
        out.write("*1s data not available.*\n\n")
        spot_c = perp_c = None

    if spot_c is not None:
        print(f"  Loaded {len(spot_c):,} aligned 1s observations")

        windows = [
            {"name": "Jan 2024 Cascade (Jan 1-5)",
             "start": "2024-01-01", "end": "2024-01-05 23:59:59",
             "role": "reference"},
            {"name": "Apr 2024 Cascade (Apr 12-18)",
             "start": "2024-04-12", "end": "2024-04-18 23:59:59",
             "role": "reference"},
            {"name": "Aug 2024 Cascade (Aug 3-9)",
             "start": "2024-08-03", "end": "2024-08-09 23:59:59",
             "role": "reference"},
        ]

    primary_params = dict(
        interval="1s",
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

            # Trade detail summary for all portfolios with trades
            for p in portfolios:
                if p.n_trades > 0:
                    wins = sum(1 for t in p.trades if t.pnl_net > 0)
                    wins_gross = sum(1 for t in p.trades if t.pnl_gross > 0)
                    losses = p.n_trades - wins
                    out.write(f"**{p.name}**: {p.n_trades} trades "
                              f"({wins}W / {losses}L, "
                              f"{wins_gross} gross winners)")
                    if p.trades:
                        out.write(f" | Best: {max(t.pnl_net for t in p.trades):.6f}"
                                  f" | Worst: {min(t.pnl_net for t in p.trades):.6f}")
                    out.write("\n")
            out.write("\n")

            # Trade log for portfolios with trades
            for p in portfolios:
                if p.n_trades > 0 and p.name != "P0_NeverTrade":
                    out.write(f"**{p.name} Trade Log (first 10):**\n\n")
                    out.write(f"| Entry | Exit | PnL (gross) | PnL (net) | Entry Reason | Exit Reason |\n")
                    out.write(f"|-------|------|-------------|-----------|--------------|-------------|\n")
                    for t in p.trades[:10]:
                        out.write(f"| {t.entry_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.exit_time.strftime('%m-%d %H:%M')} "
                                  f"| {t.pnl_gross:+.6f} "
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

    # ── 4. Price Discovery Analysis (per cascade training window) ───────
    train_windows = [w for w in windows if w["role"] == "train"]
    for tw in train_windows:
        tw_name = tw["name"]
        if tw_name not in window_rolling:
            continue
        out.write(section(f"Price Discovery Analysis — {tw_name}"))

        df = window_rolling[tw_name].to_dataframe()
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

        # Generate Figure 3 (for first training window only)
        if tw == train_windows[0]:
            print(f"\nGenerating Figure 3 for {tw_name}...")
            try:
                fig_start = tw["start"]
                fig_end = tw["end"]
                spot_day = spot_c.loc[fig_start:fig_end]
                perp_day = perp_c.loc[fig_start:fig_end]
                plot_figure3(spot_day, perp_day, df, output_dir=RESULTS_DIR,
                             title_suffix=f" — {tw_name}")
            except Exception as e:
                print(f"  Warning: Figure 3 generation failed: {e}")
                out.write(f"*Figure 3 generation failed: {e}*\n\n")

    # Price discovery analysis for each cascade window
    for w in windows:
        wname = w["name"]
        if wname not in window_rolling:
            continue
        out.write(section(f"Price Discovery — {wname}", level=3))
        df = window_rolling[wname].to_dataframe()
        out.write(f"Rolling windows: {len(df)}\n\n")

        out.write("| Statistic | d^P Spot | d^P Perp |\n")
        out.write("|-----------|----------|----------|\n")
        for stat, fn in [("Mean", "mean"), ("Std", "std"),
                         ("Min", "min"), ("Max", "max")]:
            v0 = getattr(df["d_P_0"], fn)()
            v1 = getattr(df["d_P_1"], fn)()
            out.write(f"| {stat} | {v0:.4f} | {v1:.4f} |\n")

        d_sum = df["d_P_0"] + df["d_P_1"]
        out.write(f"\nOvershooting zone (d_sum > 2): {(d_sum > 2).mean():.1%}\n\n")

    # ── 8. Write results ────────────────────────────────────────────────
    out.write("---\n\n")
    out.write("> Generated by `python3 -m scripts.run_pipeline`\n")

    RESULTS_FILE.write_text(out.getvalue())
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
