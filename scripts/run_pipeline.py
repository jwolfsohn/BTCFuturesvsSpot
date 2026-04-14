#!/usr/bin/env python3
"""
Full pipeline: load 1s data -> estimate VECM/IPES -> backtest all portfolios.

Exploits the "Misguided Price Discovery" paper's finding that futures
overshoot during liquidation cascades. IPES detects this in real-time,
enabling profitable mean-reversion strategies.

Uses 1-second BTC/USDT spot and perpetual futures from Binance.

Usage:
    python3 -m scripts.run_pipeline                   # Full: Part 1 + Part 2 + VECM
    python3 -m scripts.run_pipeline --fast            # Part 1 only (skip 1s cascades)
    python3 -m scripts.run_pipeline --no-vecm         # Stub signals, Part 1 only (seconds)

Flags:
    --fast      Skip Part 2 (1-second cascade deep-dives, Figure 3, price discovery).
                Keeps the 1-minute multi-year backtest + walk-forward intact.
    --no-vecm   Do not run the rolling VECM. Build stub signals from raw prices
                (E=0, d_P=0, cointegration_holds=False) and backtest strategies
                against them. Implies --fast. Use for rapid strategy iteration.

Outputs:
    results/resultsN.md -- Full results in Markdown (auto-increments)
    results/figure3_btc_cascade.png/pdf -- Three-panel visualization
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress numpy RuntimeWarnings from VECM numerical edge cases
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scripts.load_data import (
    load_aligned_pair, load_funding_rate,
)
from scripts.plot_paper_results import plot_figure3
from scripts.backtest import (
    run_full_pipeline,
    run_all_portfolios, results_table,
    backtest_p_mom, backtest_p_carry, backtest_p_dir,
    backtest_p_mom_a, backtest_p_mom_c, backtest_p_mom_d, backtest_p_mom_f,
    p_mom_sweep, p_carry_sweep, p_dir_sweep,
)
from scripts.vecm_ipes import WindowSignals, WINDOW_DEFAULTS

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


def _build_stub_signals(spot: pd.Series, perp: pd.Series, interval: str) -> list:
    """Build stub WindowSignals from raw prices without running VECM.

    Matches the cadence of the real pipeline (W warmup, step-sized samples)
    so downstream backtests see the same number of signals. All IPES/d^P
    fields are zero and cointegration_holds=False, which forces P_MOM onto
    its no_coint entry path and keeps P_CARRY/P_DIR's IPES gates inactive.
    """
    d = WINDOW_DEFAULTS.get(interval, WINDOW_DEFAULTS["1m"])
    W, step = d["W"], d["step"]
    both = pd.concat([spot, perp], axis=1, join="inner").dropna()
    if len(both) < W + step:
        return []
    signals = []
    zero2 = np.array([0.0, 0.0])
    for i in range(W - 1, len(both), step):
        ts = both.index[i]
        s_price = float(both.iloc[i, 0])
        p_price = float(both.iloc[i, 1])
        signals.append(WindowSignals(
            timestamp=ts,
            cointegration_holds=False,
            alpha_signs_opposite=False,
            d_P=zero2.copy(),
            E=zero2.copy(),
            ipes=zero2.copy(),
            nls=zero2.copy(),
            pils=zero2.copy(),
            covis=zero2.copy(),
            z_trans=0.0, z_perm=0.0, z_spread=0.0, z_coint_resid=0.0,
            spread=s_price - p_price,
            prices=np.array([s_price, p_price]),
            eta_P_last=0.0, eta_T_last=0.0,
            portmanteau_pvalue=1.0,
            residuals_stationary=True,
        ))
    return signals


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="BTC IPES pipeline: VECM estimation + portfolio backtests.",
    )
    p.add_argument("--fast", action="store_true",
                   help="Skip Part 2 (1-second cascade deep-dives).")
    p.add_argument("--no-vecm", dest="no_vecm", action="store_true",
                   help="Skip VECM entirely; use stub signals. Implies --fast.")
    args = p.parse_args(argv)
    if args.no_vecm and not args.fast:
        print("note: --no-vecm implies --fast; skipping Part 2.", file=sys.stderr)
        args.fast = True
    return args


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
| P_MOM_A | Ablation | P_MOM + 200d SMA | Trend-filter gate — only long above long-run SMA |
| P_MOM_C | Ablation | P_MOM + vol target | Scale leverage by inverse 30d realized vol (target 40% ann) |
| P_MOM_D | Ablation | P_MOM + DD breaker | Halve leverage at −10% DD; flatten at −20% |
| P_MOM_F | Ablation | P_MOM − IPES | IPES red-gate removed; isolates its marginal value |
| **P_CARRY** | **Always-on** | **Carry + IPES** | **Earn funding rate; IPES shields against cascade losses** |

"""


# ── Main pipeline ───────────────────────────────────────────────────────────

def main(args: argparse.Namespace | None = None):
    if args is None:
        args = _parse_args()

    RESULTS_FILE = _next_results_file()
    out = StringIO()

    mode_tag = "full"
    if args.no_vecm:
        mode_tag = "no-vecm (stub signals)"
    elif args.fast:
        mode_tag = "fast (Part 1 only)"

    out.write("# Results — BTC IPES Regime-Filtered Trading System\n\n")
    out.write(f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}\n")
    out.write(f"Mode: **{mode_tag}**\n\n")
    out.write("IPES as a real-time market quality filter for always-on BTC strategies.\n")
    out.write("1-minute multi-year backtest + 1-second cascade deep-dives.\n\n")

    out.write(STRATEGY_OVERVIEW)
    out.write("---\n")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: MULTI-YEAR BACKTEST (1-minute data, always-on strategies)
    # ══════════════════════════════════════════════════════════════════
    out.write(section("Part 1: Multi-Year Backtest (1-Minute)"))

    print("=" * 60)
    print("  PART 1: Multi-Year 1-Minute Backtest")
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
    first_date = spot_1m.index[0].strftime('%Y-%m-%d')
    last_date = spot_1m.index[-1].strftime('%Y-%m-%d')
    out.write(f"**BTC Price:** {spot_1m.iloc[0]:.0f} ({first_date}) → "
              f"{spot_1m.iloc[-1]:.0f} ({last_date}) "
              f"({(spot_1m.iloc[-1]/spot_1m.iloc[0]-1)*100:+.1f}%)\n\n")

    # Dynamic year detection — build windows from available data
    first_year = spot_1m.index[0].year
    last_year = spot_1m.index[-1].year

    fy_windows = []
    for y in range(first_year, last_year + 1):
        fy_windows.append({"name": f"H1 {y} (Jan-Jun)", "start": f"{y}-01-01",
                           "end": f"{y}-06-30 23:59:59", "role": "train"})
        fy_windows.append({"name": f"H2 {y} (Jul-Dec)", "start": f"{y}-07-01",
                           "end": f"{y}-12-31 23:59:59", "role": "test"})
        fy_windows.append({"name": f"Full Year {y}", "start": f"{y}-01-01",
                           "end": f"{y}-12-31 23:59:59", "role": "reference"})

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
        if args.no_vecm:
            stub_signals = _build_stub_signals(c1, c2, interval="1m")
            portfolios = run_all_portfolios(
                stub_signals,
                E_threshold=primary_params_1m["E_threshold"],
                tx_cost_bps=primary_params_1m["tx_cost_bps"],
            )
            metrics = results_table(portfolios)
            rolling = type("StubRolling", (), {
                "signals": stub_signals,
                "params": {"W": primary_params_1m["W"], "step": 15,
                           "interval": "1m", "stub": True},
                "to_dataframe": lambda self=None: pd.DataFrame(),
            })()
        else:
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

            # Trade log for key strategies (full-year reference windows only)
            for p in portfolios:
                if (0 < p.n_trades <= 20 and p.name in ("P_MOM", "P_CARRY", "P_DIR")
                        and w["role"] == "reference"):
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
    # WALK-FORWARD VALIDATION: Multi-year rolling
    # ══════════════════════════════════════════════════════════════════

    def classify_regime(start, end):
        """Classify a period's market regime from spot price data."""
        s = spot_1m.loc[start:end]
        if len(s) < 2:
            return "N/A"
        ret = s.iloc[-1] / s.iloc[0] - 1
        cummax = s.cummax()
        max_dd = ((s - cummax) / cummax).min()
        if max_dd < -0.30:
            return "Crash"
        elif ret > 0.20:
            return "Bull"
        elif ret < -0.20:
            return "Bear"
        else:
            return "Sideways"

    def select_best(sweep_df):
        """Select best config: Sharpe > 0 first, then highest return."""
        viable = sweep_df[sweep_df["sharpe"] > 0]
        if len(viable) > 0:
            return viable.loc[viable["total_return"].idxmax()]
        return sweep_df.loc[sweep_df["total_return"].idxmax()]

    # Build walk-forward pairs
    wf_pairs = []

    # Within-year: H1 → H2 for each year
    for y in range(first_year, last_year + 1):
        h1_key = f"H1 {y} (Jan-Jun)"
        h2_key = f"H2 {y} (Jul-Dec)"
        if h1_key in fy_signals and h2_key in fy_signals:
            wf_pairs.append((h1_key, h2_key))

    # Cross-year: Full Year N → Full Year N+1
    for y in range(first_year, last_year):
        fy_key = f"Full Year {y}"
        fy_next = f"Full Year {y + 1}"
        if fy_key in fy_signals and fy_next in fy_signals:
            wf_pairs.append((fy_key, fy_next))

    if wf_pairs:
        out.write(section("Walk-Forward Validation (Multi-Year Rolling)"))
        out.write("Parameters swept on train period, frozen, tested out-of-sample. "
                  "Within-year (H1→H2) and cross-year (Year N→Year N+1).\n\n")

        print("\n" + "=" * 60)
        print("  WALK-FORWARD VALIDATION (Multi-Year)")
        print("=" * 60)

        tx = primary_params_1m["tx_cost_bps"]
        wf_summary = []

        for train_key, test_key in wf_pairs:
            train_sigs = fy_signals[train_key]
            test_sigs = fy_signals[test_key]

            # Get test period for regime classification
            test_w = next(w for w in fy_windows if w["name"] == test_key)
            regime = classify_regime(test_w["start"], test_w["end"])

            print(f"\n  {train_key} → {test_key} ({regime})")
            row = {"train": train_key, "test": test_key, "regime": regime}

            # P_MOM sweep + test
            print(f"    P_MOM sweep...")
            h1_mom = p_mom_sweep(train_sigs, tx_cost_bps=tx, verbose=False)
            best = select_best(h1_mom)
            sel = {"ema_periods": int(best["ema_periods"]),
                   "E_threshold_red": float(best["E_threshold_red"]),
                   "dp_red_threshold": float(best["dp_red_threshold"]),
                   "ema_band_pct": float(best["ema_band_pct"])}
            wf_m = backtest_p_mom(test_sigs, **sel, tx_cost_bps=tx).metrics()
            df_m = backtest_p_mom(test_sigs, tx_cost_bps=tx).metrics()
            row["mom_wf_ret"] = wf_m["total_return"]
            row["mom_wf_sharpe"] = wf_m["sharpe"]
            row["mom_def_ret"] = df_m["total_return"]
            row["mom_def_sharpe"] = df_m["sharpe"]
            row["mom_sweep_pos"] = f"{(h1_mom['sharpe'] > 0).sum()}/{len(h1_mom)}"

            # P_CARRY sweep + test
            print(f"    P_CARRY sweep...")
            h1_carry = p_carry_sweep(train_sigs, tx_cost_bps=tx, verbose=False)
            best = select_best(h1_carry)
            sel = {"E_threshold": float(best["E_threshold"]),
                   "d_sum_threshold": float(best["d_sum_threshold"]),
                   "reentry_cooldown": int(best["reentry_cooldown"])}
            wf_c = backtest_p_carry(test_sigs, **sel, tx_cost_bps=tx).metrics()
            df_c = backtest_p_carry(test_sigs, tx_cost_bps=tx).metrics()
            row["carry_wf_ret"] = wf_c["total_return"]
            row["carry_wf_sharpe"] = wf_c["sharpe"]
            row["carry_def_ret"] = df_c["total_return"]
            row["carry_def_sharpe"] = df_c["sharpe"]

            # P_DIR sweep + test
            print(f"    P_DIR sweep...")
            h1_dir = p_dir_sweep(train_sigs, tx_cost_bps=tx, verbose=False)
            best = select_best(h1_dir)
            sel = {"dp_threshold": float(best["dp_threshold"]),
                   "stop_loss_pct": float(best["stop_loss_pct"]),
                   "trail_activation_pct": float(best["trail_activation_pct"]),
                   "max_hold_signals": int(best["max_hold_signals"])}
            wf_d = backtest_p_dir(test_sigs, **sel, tx_cost_bps=tx).metrics()
            df_d = backtest_p_dir(test_sigs, tx_cost_bps=tx).metrics()
            row["dir_wf_ret"] = wf_d["total_return"]
            row["dir_wf_sharpe"] = wf_d["sharpe"]
            row["dir_def_ret"] = df_d["total_return"]
            row["dir_def_sharpe"] = df_d["sharpe"]

            # P_MOM ablation variants on OOS test period (default params, no sweep)
            print(f"    P_MOM variants...")
            for key, fn in (("mom_a", backtest_p_mom_a),
                            ("mom_c", backtest_p_mom_c),
                            ("mom_d", backtest_p_mom_d),
                            ("mom_f", backtest_p_mom_f)):
                res = fn(test_sigs, tx_cost_bps=tx)
                m = res.metrics()
                row[f"{key}_n"] = res.n_trades
                row[f"{key}_ret"] = m["total_return"]
                row[f"{key}_sharpe"] = m["sharpe"]
                row[f"{key}_dd"] = m["max_drawdown"]

            wf_summary.append(row)

        # ── Summary tables ──────────────────────────────────────────
        out.write(section("P_MOM Walk-Forward Summary", level=3))
        out.write("| Train | Test | Regime | WF Return | WF Sharpe "
                  "| Default Return | Default Sharpe | Sweep Sharpe>0 |\n")
        out.write("|-------|------|--------|-----------|-----------|"
                  "----------------|----------------|----------------|\n")
        for r in wf_summary:
            out.write(f"| {r['train']} | {r['test']} | {r['regime']} | "
                      f"{r['mom_wf_ret']:.2%} | {r['mom_wf_sharpe']:.2f} | "
                      f"{r['mom_def_ret']:.2%} | {r['mom_def_sharpe']:.2f} | "
                      f"{r['mom_sweep_pos']} |\n")
        out.write("\n")

        out.write(section("P_CARRY Walk-Forward Summary", level=3))
        out.write("| Train | Test | Regime | WF Return | WF Sharpe "
                  "| Default Return | Default Sharpe |\n")
        out.write("|-------|------|--------|-----------|-----------|"
                  "----------------|----------------|\n")
        for r in wf_summary:
            out.write(f"| {r['train']} | {r['test']} | {r['regime']} | "
                      f"{r['carry_wf_ret']:.2%} | {r['carry_wf_sharpe']:.2f} | "
                      f"{r['carry_def_ret']:.2%} | {r['carry_def_sharpe']:.2f} |\n")
        out.write("\n")

        out.write(section("P_DIR Walk-Forward Summary", level=3))
        out.write("| Train | Test | Regime | WF Return | WF Sharpe "
                  "| Default Return | Default Sharpe |\n")
        out.write("|-------|------|--------|-----------|-----------|"
                  "----------------|----------------|\n")
        for r in wf_summary:
            out.write(f"| {r['train']} | {r['test']} | {r['regime']} | "
                      f"{r['dir_wf_ret']:.2%} | {r['dir_wf_sharpe']:.2f} | "
                      f"{r['dir_def_ret']:.2%} | {r['dir_def_sharpe']:.2f} |\n")
        out.write("\n")

        # ── P_MOM ablation variant summaries (OOS, default params) ──
        variant_info = [
            ("P_MOM_A", "mom_a", "200-day SMA trend gate"),
            ("P_MOM_C", "mom_c", "Vol-targeted leverage (40% ann, 30d lookback)"),
            ("P_MOM_D", "mom_d", "Drawdown circuit breaker (−10% halve / −20% halt)"),
            ("P_MOM_F", "mom_f", "IPES ablation — pure EMA crossover"),
        ]
        for title, key, desc in variant_info:
            out.write(section(f"{title} Walk-Forward Summary", level=3))
            out.write(f"*{desc}. Default params on OOS test period (no sweep).*\n\n")
            out.write("| Train | Test | Regime | Trades | Return | Sharpe | Max DD |\n")
            out.write("|-------|------|--------|--------|--------|--------|--------|\n")
            for r in wf_summary:
                out.write(f"| {r['train']} | {r['test']} | {r['regime']} | "
                          f"{r[f'{key}_n']} | "
                          f"{r[f'{key}_ret']:.2%} | "
                          f"{r[f'{key}_sharpe']:.2f} | "
                          f"{r[f'{key}_dd']:.2%} |\n")
            out.write("\n")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: CASCADE DEEP-DIVES (1-second data)
    # ══════════════════════════════════════════════════════════════════
    if args.fast:
        out.write("---\n\n")
        out.write("> Generated by `python3 -m scripts.run_pipeline "
                  f"{'--no-vecm' if args.no_vecm else '--fast'}`\n")
        RESULTS_FILE.write_text(out.getvalue())
        print(f"\n[{mode_tag}] Results written to {RESULTS_FILE}")
        return

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
    main(_parse_args())
