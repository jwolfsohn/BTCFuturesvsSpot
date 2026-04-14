"""
Walk-forward backtesting engine for BTC spot vs perpetual futures.

Consumes signals from vecm_ipes.rolling_estimate() and applies the
decision logic adapted for BTC spot vs BTC perpetual futures.

Retained portfolios (chosen ex-ante to avoid multiple-comparisons overfitting):
    P0  -- Never Trade (null benchmark)
    P2A -- Standard Cointegration (academic benchmark)
    P3  -- Cointegration + IPES Filter (tests whether IPES overshooting
            detection predicts mean reversion — a project hypothesis,
            not a claim from the paper)

Key adaptations from stablecoin reference:
    - _identify_dislocation_leg() replaces _identify_depeg_leg()
    - Percentage-based stop-losses instead of dollar-level
    - Max holding period to prevent funding rate bleed
    - Updated defaults for BTC price levels
"""
from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass
from math import log
from typing import Optional

import numpy as np
import pandas as pd

from scripts.vecm_ipes import (
    rolling_estimate, run_spot_perp_estimation,
)


# ── Configuration ────────────────────────────────────────────────────────────

# Default thresholds (to be swept in robustness)
DEFAULT_E_THRESHOLD = 0.5
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_EXIT = 0.5           # "reverts to mean" -- tolerance band for discrete stepping
DEFAULT_Z_PERM_MAX = 1.5       # max permanent z-score for P3 entry
DEFAULT_Z_PERM_SPIKE = 2.5     # permanent z-score exit trigger
DEFAULT_STOP_LOSS_PCT = 0.02   # 2% adverse move (percentage-based for BTC)
DEFAULT_TX_COST_BPS = 5        # ~3 bps futures + ~2 bps spot at taker rates
DEFAULT_TX_COST_BPS_MAKER = 2  # ~1 bps futures + ~1 bps spot at maker rates
DEFAULT_LEVERAGE = 10           # conservative leverage for BTC perp pairs trade
DEFAULT_MIN_SPREAD_EXIT = 1.0  # min absolute |delta_spread| in dollars ($1 on BTC)
DEFAULT_MAX_HOLD_SIGNALS = 288  # 288 signals x 5-min step = 24 hours (1s data)

# Stop-loss grid for P4 (percentage-based)
P4_STOP_LEVELS_PCT = [0.005, 0.01, 0.02, 0.03, 0.05]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    """A single completed round-trip trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_spread: float
    exit_spread: float
    direction: int           # +1 = long spread, -1 = short spread
    entry_reason: str
    exit_reason: str
    pnl_gross: float         # before tx costs
    pnl_net: float           # after tx costs


@dataclass
class PortfolioResult:
    """Backtest results for a single portfolio configuration."""
    name: str
    trades: list
    equity_curve: pd.Series
    params: dict

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def metrics(self) -> dict:
        """Compute performance metrics."""
        if len(self.trades) == 0:
            return {
                "portfolio": self.name,
                "n_trades": 0,
                "total_return": 0.0,
                "ann_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_trade_pnl_gross": 0.0,
                "win_rate": 0.0,
                "gross_win_rate": 0.0,
                "breakeven_tc_bps": 0.0,
            }

        eq = self.equity_curve
        returns = eq.pct_change().dropna()

        # Total and annualized return
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0

        # Estimate trading days from the data timespan
        if len(eq) > 1:
            span_days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400
            ann_factor = 365.25 / max(span_days, 1)
        else:
            ann_factor = 1.0
        ann_ret = (1 + total_ret) ** ann_factor - 1.0

        # Sharpe (annualized)
        if len(returns) > 1 and returns.std() > 0:
            dt_median = eq.index.to_series().diff().median()
            if dt_median and dt_median.total_seconds() > 0:
                periods_per_year = 365.25 * 86400 / dt_median.total_seconds()
            else:
                periods_per_year = 252 * 24  # fallback: hourly
            sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Trade-level stats
        pnls = [t.pnl_net for t in self.trades]
        pnls_gross = [t.pnl_gross for t in self.trades]
        avg_pnl = np.mean(pnls) if pnls else 0.0
        avg_pnl_gross = np.mean(pnls_gross) if pnls_gross else 0.0
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
        gross_win_rate = sum(1 for p in pnls_gross if p > 0) / len(pnls_gross) if pnls_gross else 0.0
        # Breakeven TC: the max combined one-way cost (bps) at which avg gross PnL = 0
        # avg_net = avg_gross + tc; breakeven when tc = -avg_gross
        # tc = ln((1-C)/(1+C)) ≈ -2C → C ≈ avg_gross / 2 → bps = C * 10000
        breakeven_tc_bps = 0.0
        if avg_pnl_gross > 0:
            breakeven_tc_bps = (avg_pnl_gross / 2) * 10_000

        return {
            "portfolio": self.name,
            "n_trades": len(self.trades),
            "total_return": total_ret,
            "ann_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "avg_trade_pnl": avg_pnl,
            "avg_trade_pnl_gross": avg_pnl_gross,
            "win_rate": win_rate,
            "gross_win_rate": gross_win_rate,
            "breakeven_tc_bps": breakeven_tc_bps,
        }


# ── Transaction cost ─────────────────────────────────────────────────────────

def round_trip_cost(cost_bps: float) -> float:
    """
    Round-trip transaction cost as a log-return drag.

    cost_bps is the combined one-way cost for both legs of the pair
    (e.g., 5 bps = ~3 bps futures + ~2 bps spot).
    Round trip = 2 * one-way cost.
    Formula: ln((1-C)/(1+C)) where C = cost_bps / 10000.
    For small C this approximates -2C.
    """
    C = cost_bps / 10_000
    if C >= 1.0:
        return -np.inf
    return log((1 - C) / (1 + C))


# ── Portfolio backtesting engines ────────────────────────────────────────────

def _identify_dislocation_leg(prices: np.ndarray) -> int:
    """
    Index (0=spot, 1=perp) of the market driving the basis dislocation.

    For BTC spot vs perp, "dislocation" = which price has deviated more
    from the pair's midpoint. Unlike stablecoins (deviation from $1),
    we measure deviation from the mean of the two log-prices.
    """
    log_p = np.log(prices)
    mid = log_p.mean()
    dev = np.abs(log_p - mid)
    return int(np.argmax(dev))


def _close_position(direction, entry_spread, entry_time, entry_reason,
                     s, tc, trades, equity, timestamps, exit_reason,
                     entry_mid_price=None, leverage=1):
    """Helper to close a position and record the trade.

    PnL is normalized by entry_mid_price to produce a return (not a dollar amount).
    Leverage amplifies returns on capital: a 10x leveraged position turns a 0.02%
    spread return into a 0.2% return on capital. TX cost also scales with leverage
    since the full position size (leverage * capital) pays the fee.
    """
    pnl_gross_raw = direction * (s.spread - entry_spread)
    # Normalize to return: divide by entry midpoint price, then apply leverage
    if entry_mid_price and entry_mid_price > 0:
        pnl_gross = pnl_gross_raw / entry_mid_price * leverage
    else:
        pnl_gross = pnl_gross_raw * leverage
    pnl_net = pnl_gross + tc * leverage
    trades.append(Trade(
        entry_time=entry_time, exit_time=s.timestamp,
        entry_spread=entry_spread, exit_spread=s.spread,
        direction=direction, entry_reason=entry_reason,
        exit_reason=exit_reason,
        pnl_gross=pnl_gross, pnl_net=pnl_net,
    ))
    equity.append(equity[-1] * (1 + pnl_net))
    timestamps.append(s.timestamp)


def backtest_p1(
    signals: list,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    has_stop_loss: bool = False,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
) -> PortfolioResult:
    """
    Portfolio 1 -- Distance Method.
    Entry: |z_spread| > z_entry.
    Exit: |z_spread| <= z_exit AND |delta_spread| >= min_spread_exit.
    Sub-test A: no stop-loss. Sub-test B: percentage-based stop-loss.
    """
    name = f"P1{'B' if has_stop_loss else 'A'}"
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0

    for s in signals:
        if not in_position:
            if abs(s.z_spread) > z_entry:
                in_position = True
                direction = -1 if s.z_spread > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"z_spread={s.z_spread:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None
            spread_moved = abs(s.spread - entry_spread)

            # Max holding period
            if signals_held >= max_hold_signals:
                exit_reason = f"max_holding_period ({signals_held} signals)"
            elif abs(s.z_spread) <= z_exit and spread_moved >= min_spread_exit:
                exit_reason = f"z-score reverted (ds={spread_moved:.2f})"
            elif has_stop_loss:
                current_mid = (s.prices[0] + s.prices[1]) / 2
                pct_change = (current_mid - entry_mid_price) / entry_mid_price
                if abs(pct_change) > stop_loss_pct and (pct_change * direction) < 0:
                    exit_reason = f"stop-loss at {pct_change*100:+.2f}%"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"z_entry": z_entry, "z_exit": z_exit,
                                   "stop_loss_pct": stop_loss_pct if has_stop_loss else None,
                                   "tx_cost_bps": tx_cost_bps,
                                   "min_spread_exit": min_spread_exit,
                                   "max_hold_signals": max_hold_signals})


def backtest_p2(
    signals: list,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    has_stop_loss: bool = False,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
) -> PortfolioResult:
    """
    Portfolio 2 -- Standard Cointegration.
    Uses cointegrating residual z-score (beta'log_p) instead of raw spread.
    Cointegration break halts trading.
    """
    name = f"P2{'B' if has_stop_loss else 'A'}"
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0

    for s in signals:
        # Emergency: cointegration break
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue  # skip all trading when cointegration is broken

        if not in_position:
            if abs(s.z_coint_resid) > z_entry:
                in_position = True
                direction = -1 if s.z_coint_resid > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"z_coint={s.z_coint_resid:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None
            spread_moved = abs(s.spread - entry_spread)

            # Max holding period
            if signals_held >= max_hold_signals:
                exit_reason = f"max_holding_period ({signals_held} signals)"
            elif abs(s.z_coint_resid) <= z_exit and spread_moved >= min_spread_exit:
                exit_reason = f"z-score reverted (ds={spread_moved:.2f})"
            elif has_stop_loss:
                current_mid = (s.prices[0] + s.prices[1]) / 2
                pct_change = (current_mid - entry_mid_price) / entry_mid_price
                if abs(pct_change) > stop_loss_pct and (pct_change * direction) < 0:
                    exit_reason = f"stop-loss at {pct_change*100:+.2f}%"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"z_entry": z_entry, "z_exit": z_exit,
                                   "stop_loss_pct": stop_loss_pct if has_stop_loss else None,
                                   "tx_cost_bps": tx_cost_bps,
                                   "min_spread_exit": min_spread_exit,
                                   "max_hold_signals": max_hold_signals})


def backtest_p3(
    signals: list,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    z_perm_max: float = DEFAULT_Z_PERM_MAX,
    z_perm_spike: float = DEFAULT_Z_PERM_SPIKE,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
    name: str = "P3",
    use_perverse_block: bool = True,
    use_bilateral_block: bool = True,
    use_dp_overshoot: bool = True,
    use_ei_threshold: bool = True,
    use_alpha_diverging_block: bool = True,
    use_z_perm_max_block: bool = True,
    use_z_trans_entry: bool = True,
    stability_lookback: int = 0,    # 0 = disabled; N > 0 = require N consecutive stable bars pre-entry
) -> PortfolioResult:
    """
    Portfolio 3 -- Cointegration + IPES Filter (core contribution).

    Entry requires ALL of:
      1. Cointegration holds
      2. alpha signs opposite
      3. d^P >= 0 for both (no perverse response)
      4. NOT bilateral high E_i (systemic stress block)
      5. |z_trans| > z_entry (fresh transitory dislocation)
      6. |z_perm| < z_perm_max (not permanent repricing)
      7. d^P > 1 for dislocation leg (confirmed overshooting)
      8. E_i > threshold for dislocation leg

    Exit on ANY of:
      1. z_trans reverts to 0 AND |delta_spread| >= min_spread_exit
      2. z_perm spikes (bypasses spread check -- emergency)
      3. E_i regime flip (bypasses spread check -- structural)
      4. d^P < 0 (emergency)
      5. Cointegration breaks (emergency)
      6. Max holding period exceeded
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    prev_E = None  # for regime flip detection
    signals_held = 0
    stab_deque = collections.deque(maxlen=max(stability_lookback, 1))

    for s in signals:
        # ── Emergency exits (always checked) ─────────────────────────
        emergency_exit = None
        if not s.cointegration_holds:
            emergency_exit = "cointegration_break"
        elif use_alpha_diverging_block and not s.alpha_signs_opposite:
            emergency_exit = "alpha_diverging"
        elif use_perverse_block and (s.d_P[0] < 0 or s.d_P[1] < 0):
            emergency_exit = f"perverse_response d_P=[{s.d_P[0]:.2f},{s.d_P[1]:.2f}]"

        if emergency_exit and in_position:
            _close_position(direction, entry_spread, entry_time, entry_reason,
                            s, tc, trades, equity, timestamps, emergency_exit,
                            entry_mid_price=entry_mid_price)
            in_position = False

        if emergency_exit:
            stab_deque.append(False)    # cointegration/alpha broken -- poison window
            prev_E = s.E.copy()
            continue

        # ── Pre-entry stability check ─────────────────────────────────
        if (stability_lookback > 0
                and not in_position
                and not (len(stab_deque) == stability_lookback and all(stab_deque))):
            stab_deque.append(s.cointegration_holds and s.alpha_signs_opposite)
            prev_E = s.E.copy()
            continue

        dislo = _identify_dislocation_leg(s.prices)

        if not in_position:
            # ── Full entry gate ──────────────────────────────────────
            # Layer 3: Bilateral E_i block
            if use_bilateral_block and (s.E[0] > E_threshold and s.E[1] > E_threshold):
                prev_E = s.E.copy()
                continue  # systemic stress -- block

            # Layer 4: E_i + d^P > 1
            if use_ei_threshold and (s.E[dislo] <= E_threshold):
                prev_E = s.E.copy()
                continue  # pricing is accurate -- don't trade

            if use_dp_overshoot and (s.d_P[dislo] <= 1.0):
                prev_E = s.E.copy()
                continue  # under-reaction or accurate

            # Layer 5: Shock z-scores
            if use_z_trans_entry and abs(s.z_trans) <= z_entry:
                prev_E = s.E.copy()
                continue  # no fresh transitory dislocation

            if use_z_perm_max_block and abs(s.z_perm) >= z_perm_max:
                prev_E = s.E.copy()
                continue  # permanent repricing in progress

            # All gates passed -- enter
            in_position = True
            direction = -1 if s.z_trans > 0 else 1
            entry_spread = s.spread
            entry_time = s.timestamp
            entry_reason = (f"d_P={s.d_P[dislo]:.2f} E={s.E[dislo]:.3f} "
                           f"z_T={s.z_trans:.2f} z_P={s.z_perm:.2f}")
            entry_mid_price = (s.prices[0] + s.prices[1]) / 2
            signals_held = 0
        else:
            # ── Exit conditions (any triggers exit) ──────────────────
            signals_held += 1
            exit_reason = None

            # Exit 6: Max holding period
            if signals_held >= max_hold_signals:
                exit_reason = f"max_holding_period ({signals_held} signals)"

            # Exit 2: Permanent shock spike
            elif abs(s.z_perm) >= z_perm_spike:
                exit_reason = f"perm_shock_spike z_P={s.z_perm:.2f}"

            # Exit 3: E_i regime flip
            elif (prev_E is not None and
                  prev_E[dislo] > E_threshold and
                  s.E[dislo] <= E_threshold):
                exit_reason = "E_i_regime_flip"

            # Exit 1: Transitory z-score reverts AND spread has actually moved
            elif abs(s.z_trans) <= z_exit:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted={s.z_trans:.2f} (ds={spread_moved:.2f})"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

        prev_E = s.E.copy()
        stab_deque.append(s.cointegration_holds and s.alpha_signs_opposite)

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"E_threshold": E_threshold,
                                   "z_entry": z_entry, "z_exit": z_exit,
                                   "z_perm_max": z_perm_max,
                                   "z_perm_spike": z_perm_spike,
                                   "tx_cost_bps": tx_cost_bps,
                                   "min_spread_exit": min_spread_exit,
                                   "max_hold_signals": max_hold_signals,
                                   "stability_lookback": stability_lookback})


def backtest_p4(
    signals: list,
    stop_loss_pct: float = 0.02,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
) -> PortfolioResult:
    """
    Portfolio 4 -- Cointegration + Standard Stop-Loss.
    Same as P2A but with configurable percentage-based stop-loss.
    Tests whether stop-losses help or hurt (literature: they hurt).
    """
    name = f"P4_SL{stop_loss_pct*100:.1f}pct"
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        if not in_position:
            if abs(s.z_coint_resid) > z_entry:
                in_position = True
                direction = -1 if s.z_coint_resid > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"z_coint={s.z_coint_resid:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None
            spread_moved = abs(s.spread - entry_spread)

            # Max holding period
            if signals_held >= max_hold_signals:
                exit_reason = f"max_holding_period ({signals_held} signals)"
            elif abs(s.z_coint_resid) <= z_exit and spread_moved >= min_spread_exit:
                exit_reason = f"z-score reverted (ds={spread_moved:.2f})"
            else:
                # Percentage-based stop-loss
                current_mid = (s.prices[0] + s.prices[1]) / 2
                pct_change = (current_mid - entry_mid_price) / entry_mid_price
                if abs(pct_change) > stop_loss_pct and (pct_change * direction) < 0:
                    exit_reason = f"stop-loss at {pct_change*100:+.2f}%"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"stop_loss_pct": stop_loss_pct,
                                   "z_entry": z_entry, "z_exit": z_exit,
                                   "tx_cost_bps": tx_cost_bps,
                                   "min_spread_exit": min_spread_exit,
                                   "max_hold_signals": max_hold_signals})


# ── New strategies exploiting price discovery insights ──────────────────────
#
# Key improvements over P1-P4 benchmarks:
#   1. Leverage (10x default) — realistic for BTC perp pairs trades
#   2. Maker tx cost (2 bps) — Binance maker rates, not taker
#   3. Spread-direction filter — only enter AFTER the spread peaks (post-peak fade)
#   4. Profit-based exits — only close on z-score reversion if trade is profitable
#   5. Stop-loss on gross PnL — cut losses at configurable threshold


def backtest_p5(
    signals: list,
    dp_threshold: float = 1.2,
    z_trans_min: float = 0.5,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS_MAKER,
    leverage: int = DEFAULT_LEVERAGE,
    max_hold_signals: int = 48,
    stop_loss_spread_pct: float = 0.01,
    name: str = "P5_OvershootFade",
) -> PortfolioResult:
    """
    Portfolio 5 -- Post-Peak Overshooting Fade.

    Exploits the paper's core finding: overshooting reverts. Enters AFTER
    the spread starts retreating (post-peak filter). Exits only when the
    fundamental overshooting signal resolves (d^P falls below 1.0) AND
    the trade is profitable. No z-score-based exits — these cause
    premature closes from window-rolling noise.

    Entry:
      - Cointegration holds
      - d^P[leg] > dp_threshold (confirmed overshooting)
      - |z_trans| > z_trans_min (transitory dislocation)
      - Spread is retreating (post-peak filter)
    Exit:
      - Profitable + d^P < 1.0 (fundamental: overshooting resolved)
      - Stop-loss: adverse spread move > stop_loss_spread_pct
      - Max hold exceeded
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0
    entry_overshoot_leg = 0
    prev_spread = None

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            prev_spread = s.spread
            continue

        if not in_position:
            overshoot_leg = -1
            if s.d_P[0] > dp_threshold and s.d_P[1] > dp_threshold:
                overshoot_leg = 0 if s.d_P[0] > s.d_P[1] else 1
            elif s.d_P[0] > dp_threshold:
                overshoot_leg = 0
            elif s.d_P[1] > dp_threshold:
                overshoot_leg = 1

            spread_retreating = False
            if prev_spread is not None:
                if s.z_trans > 0:
                    spread_retreating = s.spread < prev_spread
                else:
                    spread_retreating = s.spread > prev_spread

            if overshoot_leg >= 0 and abs(s.z_trans) > z_trans_min and spread_retreating:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = (f"postpeak d_P[{overshoot_leg}]={s.d_P[overshoot_leg]:.2f} "
                               f"z_T={s.z_trans:.2f}")
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                entry_overshoot_leg = overshoot_leg
                signals_held = 0
        else:
            signals_held += 1
            gross_pnl = direction * (s.spread - entry_spread) / entry_mid_price
            exit_reason = None

            if gross_pnl < -stop_loss_spread_pct:
                exit_reason = f"stop_loss gross={gross_pnl*100:.3f}%"
            elif signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} sigs) gross={gross_pnl*100:.3f}%"
            # Fundamental exit: overshooting resolved AND profitable
            elif gross_pnl > 0 and s.d_P[entry_overshoot_leg] < 1.0:
                exit_reason = f"profitable_resolved gross={gross_pnl*100:.3f}%"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False

        prev_spread = s.spread

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_threshold": dp_threshold,
                                   "z_trans_min": z_trans_min,
                                   "tx_cost_bps": tx_cost_bps,
                                   "leverage": leverage,
                                   "max_hold_signals": max_hold_signals})


def backtest_p6(
    signals: list,
    dp_sum_entry: float = 2.0,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS_MAKER,
    leverage: int = DEFAULT_LEVERAGE,
    max_hold_signals: int = 48,
    stop_loss_spread_pct: float = 0.01,
    name: str = "P6_ConflictFade",
) -> PortfolioResult:
    """
    Portfolio 6 -- Post-Peak Conflict Zone Fade.

    When d^P_0 + d^P_1 > 2 (overshooting zone), both markets overreacted.
    Enters AFTER the spread peaks. Exits when the fundamental signal
    resolves (d_sum returns to reliable zone) AND trade is profitable.
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0
    prev_spread = None

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            prev_spread = s.spread
            continue

        d_sum = s.d_P[0] + s.d_P[1]

        if not in_position:
            spread_retreating = False
            if prev_spread is not None:
                if s.z_trans > 0:
                    spread_retreating = s.spread < prev_spread
                else:
                    spread_retreating = s.spread > prev_spread

            if d_sum > dp_sum_entry and spread_retreating:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"conflict d_sum={d_sum:.2f} z_T={s.z_trans:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            gross_pnl = direction * (s.spread - entry_spread) / entry_mid_price
            exit_reason = None

            if gross_pnl < -stop_loss_spread_pct:
                exit_reason = f"stop_loss gross={gross_pnl*100:.3f}%"
            elif signals_held >= max_hold_signals:
                exit_reason = f"max_hold gross={gross_pnl*100:.3f}%"
            # Fundamental exit: conflict zone resolved AND profitable
            elif gross_pnl > 0 and d_sum < 1.5:
                exit_reason = f"profitable_resolved d_sum={d_sum:.2f}"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False

        prev_spread = s.spread

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_sum_entry": dp_sum_entry,
                                   "tx_cost_bps": tx_cost_bps,
                                   "leverage": leverage,
                                   "max_hold_signals": max_hold_signals})


def backtest_p7(
    signals: list,
    z_entry: float = 1.5,
    hold_signals: int = 6,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS_MAKER,
    leverage: int = DEFAULT_LEVERAGE,
    name: str = "P7_Momentum",
) -> PortfolioResult:
    """
    Portfolio 7 -- Cascade Momentum.

    RIDES the cascade instead of fading it. When a transitory shock spikes,
    the spread continues in that direction for several signals before reverting.
    Goes WITH the momentum for a fixed short hold, then exits.

    Entry: |z_trans| > z_entry AND cointegration holds
    Direction: WITH the shock (opposite of mean-reversion strategies)
    Exit: Fixed hold of N signals (time-based, no signal-based exit)
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            continue

        if not in_position:
            if abs(s.z_trans) > z_entry:
                in_position = True
                # MOMENTUM: go WITH the shock direction (opposite of fade)
                direction = +1 if s.z_trans > 0 else -1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"momentum z_T={s.z_trans:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            if signals_held >= hold_signals:
                gross_pnl = direction * (s.spread - entry_spread) / entry_mid_price
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps,
                                f"fixed_exit ({hold_signals} sigs) gross={gross_pnl*100:.3f}%",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"z_entry": z_entry, "hold_signals": hold_signals,
                                   "tx_cost_bps": tx_cost_bps, "leverage": leverage})


def backtest_p8(
    signals: list,
    E_low: float = 0.3,
    E_high: float = 0.7,
    z_entry_efficient: float = 2.0,
    z_entry_mid: float = 1.0,
    z_entry_stressed: float = 0.5,
    z_perm_max: float = DEFAULT_Z_PERM_MAX,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS_MAKER,
    leverage: int = DEFAULT_LEVERAGE,
    max_hold_signals: int = 24,
    stop_loss_spread_pct: float = 0.005,
    name: str = "P8_IPESAdaptive",
) -> PortfolioResult:
    """
    Portfolio 8 -- IPES Adaptive Entry with Profit Exits.

    P3-style entry gates but adapts z_entry based on pricing error regime,
    uses leverage, and only exits on signal reversion when profitable.

    Regime:
      - Efficient (E < E_low):  z_entry = 2.0 (conservative)
      - Mid (E_low <= E < E_high): z_entry = 1.0
      - Stressed (E >= E_high): z_entry = 0.5 (aggressive)
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0
    prev_spread = None

    for s in signals:
        if not s.cointegration_holds or not s.alpha_signs_opposite:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "emergency",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            prev_spread = s.spread
            continue

        if s.d_P[0] < 0 or s.d_P[1] < 0:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "perverse",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            prev_spread = s.spread
            continue

        # Adaptive z_entry
        max_E = max(s.E[0], s.E[1])
        if max_E >= E_high:
            adaptive_z = z_entry_stressed
        elif max_E >= E_low:
            adaptive_z = z_entry_mid
        else:
            adaptive_z = z_entry_efficient

        dislo = _identify_dislocation_leg(s.prices)

        if not in_position:
            if s.E[0] > 0.5 and s.E[1] > 0.5:
                prev_spread = s.spread
                continue
            if s.d_P[dislo] <= 1.0:
                prev_spread = s.spread
                continue
            if abs(s.z_trans) <= adaptive_z:
                prev_spread = s.spread
                continue
            if abs(s.z_perm) >= z_perm_max:
                prev_spread = s.spread
                continue

            spread_retreating = False
            if prev_spread is not None:
                if s.z_trans > 0:
                    spread_retreating = s.spread < prev_spread
                else:
                    spread_retreating = s.spread > prev_spread

            if spread_retreating:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = (f"adaptive z={adaptive_z:.1f} E_max={max_E:.2f} "
                               f"d_P={s.d_P[dislo]:.2f}")
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            gross_pnl = direction * (s.spread - entry_spread) / entry_mid_price
            exit_reason = None

            if gross_pnl < -stop_loss_spread_pct:
                exit_reason = f"stop_loss gross={gross_pnl*100:.3f}%"
            elif signals_held >= max_hold_signals:
                exit_reason = f"max_hold gross={gross_pnl*100:.3f}%"
            # Fundamental exit: d^P resolved for dislocation leg AND profitable
            elif gross_pnl > 0 and s.d_P[dislo] < 1.0:
                exit_reason = f"profitable_resolved gross={gross_pnl*100:.3f}%"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False

        prev_spread = s.spread

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"E_low": E_low, "E_high": E_high,
                                   "tx_cost_bps": tx_cost_bps,
                                   "leverage": leverage,
                                   "max_hold_signals": max_hold_signals})


def backtest_p9(
    signals: list,
    dp_perp_min: float = 1.3,
    E_perp_min: float = 0.3,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS_MAKER,
    leverage: int = DEFAULT_LEVERAGE,
    max_hold_signals: int = 48,
    stop_loss_spread_pct: float = 0.01,
    name: str = "P9_CascadeFade",
) -> PortfolioResult:
    """
    Portfolio 9 -- Post-Peak Cascade Fade.

    Targets liquidation cascade patterns where futures overreact.
    Enters AFTER the spread peaks during the cascade. Holds through
    the full reversion with a generous stop-loss.

    Entry:
      - Cointegration holds
      - d^P_perp > dp_perp_min (perp overshooting)
      - E_perp > E_perp_min (perp pricing error)
      - Spread is falling (post-peak)
    Exit:
      - Profitable + d^P_perp < 1.0 (fundamental: overshooting resolved)
      - Stop-loss / max hold
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0
    prev_spread = None

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False
            prev_spread = s.spread
            continue

        if not in_position:
            # Post-peak filter
            spread_retreating = False
            if prev_spread is not None:
                if s.z_trans > 0:
                    spread_retreating = s.spread < prev_spread
                else:
                    spread_retreating = s.spread > prev_spread

            if s.d_P[1] > dp_perp_min and s.E[1] > E_perp_min and spread_retreating:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = (f"cascade d_P_perp={s.d_P[1]:.2f} "
                               f"E_perp={s.E[1]:.3f}")
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            gross_pnl = direction * (s.spread - entry_spread) / entry_mid_price
            exit_reason = None

            if gross_pnl < -stop_loss_spread_pct:
                exit_reason = f"stop_loss gross={gross_pnl*100:.3f}%"
            elif signals_held >= max_hold_signals:
                exit_reason = f"max_hold gross={gross_pnl*100:.3f}%"
            # Fundamental exit: perp overshooting resolved AND profitable
            elif gross_pnl > 0 and s.d_P[1] < 1.0:
                exit_reason = f"profitable_resolved d_P={s.d_P[1]:.2f}"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price, leverage=leverage)
                in_position = False

        prev_spread = s.spread

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_perp_min": dp_perp_min,
                                   "E_perp_min": E_perp_min,
                                   "tx_cost_bps": tx_cost_bps,
                                   "leverage": leverage,
                                   "max_hold_signals": max_hold_signals})


# ── New portfolios: directional and regime-based ─────────────────────────────


def backtest_p_bh(signals: list) -> PortfolioResult:
    """
    Buy & Hold BTC benchmark.

    Enters long BTC (via spot) at first signal and holds for the entire period.
    This is the relevant directional benchmark since BTC spot-perp strategies
    involve BTC exposure (unlike stablecoin pairs which are USD-neutral).
    """
    if not signals:
        eq = pd.Series([1.0], index=[pd.Timestamp.now(tz="UTC")])
        return PortfolioResult(name="P_BH", trades=[], equity_curve=eq, params={})

    start_price = signals[0].prices[0]  # spot price at first signal
    timestamps = []
    equity = []
    for s in signals:
        timestamps.append(s.timestamp)
        equity.append(s.prices[0] / start_price)  # normalized to 1.0

    eq_series = pd.Series(equity, index=timestamps)
    # Record as a single trade spanning the period
    total_ret = equity[-1] - 1.0
    trades = [Trade(
        entry_time=signals[0].timestamp,
        exit_time=signals[-1].timestamp,
        entry_spread=0, exit_spread=0,
        direction=1, entry_reason="buy_and_hold",
        exit_reason="end_of_period",
        pnl_gross=total_ret, pnl_net=total_ret,
    )]
    return PortfolioResult(name="P_BH", trades=trades, equity_curve=eq_series,
                           params={})


def backtest_p_dir(
    signals: list,
    dp_threshold: float = 1.2,
    stop_loss_pct: float = 0.01,
    trail_activation_pct: float = 0.01,
    trail_fraction: float = 0.5,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    leverage: int = 5,
    max_hold_signals: int = 288,
    name: str = "P_DIR",
) -> PortfolioResult:
    """
    Directional Cascade Fade using IPES overshooting detection.

    When IPES detects perp overshooting (d^P_perp > threshold), the perp has
    moved too far relative to fundamentals. Take an UNHEDGED directional
    position betting on perp recovery:
      - If perp overshot downward (spot > perp): LONG BTC via perp
      - If perp overshot upward (perp > spot): SHORT BTC via perp

    This is DIRECTIONAL, not spread-neutral. Returns scale with BTC price
    volatility (1-5%/day), not basis volatility (~5 bps).

    Risk management:
      - Fixed stop-loss at stop_loss_pct (default 1% price move = 5% on capital)
      - Trailing stop: once unrealized gain reaches trail_activation_pct,
        stop trails at trail_fraction of peak unrealized gain

    This tests whether IPES overshooting detection predicts directional
    mean reversion. This is a project hypothesis, not a paper claim.
    """
    tc_one_way = round_trip_cost(tx_cost_bps) / 2  # half the round trip for one leg
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_price = 0.0
    entry_time = None
    entry_reason = ""
    signals_held = 0
    prev_spread = None
    peak_pnl = 0.0  # track best unrealized PnL for trailing stop

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                exit_price = s.prices[1]  # perp price
                pnl_gross = direction * (exit_price - entry_price) / entry_price * leverage
                pnl_net = pnl_gross + tc_one_way * leverage
                trades.append(Trade(
                    entry_time=entry_time, exit_time=s.timestamp,
                    entry_spread=entry_price, exit_spread=exit_price,
                    direction=direction, entry_reason=entry_reason,
                    exit_reason="cointegration_break",
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                ))
                equity.append(equity[-1] * (1 + pnl_net))
                timestamps.append(s.timestamp)
                in_position = False
            prev_spread = s.spread
            continue

        if not in_position:
            # Entry: perp overshooting detected
            if s.d_P[1] > dp_threshold:
                # Spread retreating = post-peak filter
                spread_retreating = False
                if prev_spread is not None:
                    if s.spread > 0:  # spot > perp (perp overshot down)
                        spread_retreating = s.spread < prev_spread
                    else:  # perp > spot (perp overshot up)
                        spread_retreating = s.spread > prev_spread

                if spread_retreating:
                    in_position = True
                    # If spot > perp: perp overshot DOWN → LONG perp (buy BTC)
                    # If perp > spot: perp overshot UP → SHORT perp (sell BTC)
                    direction = 1 if s.spread > 0 else -1
                    entry_price = s.prices[1]  # perp price
                    entry_time = s.timestamp
                    entry_reason = (f"cascade d_P_perp={s.d_P[1]:.2f} "
                                   f"spread={s.spread:.1f}")
                    signals_held = 0
                    peak_pnl = 0.0

                    # Entry cost
                    equity.append(equity[-1] * (1 + tc_one_way * leverage))
                    timestamps.append(s.timestamp)
        else:
            signals_held += 1
            current_price = s.prices[1]  # perp price
            pnl_unrealized = direction * (current_price - entry_price) / entry_price * leverage
            exit_reason = None

            # Update peak unrealized PnL for trailing stop
            if pnl_unrealized > peak_pnl:
                peak_pnl = pnl_unrealized

            # Fixed stop-loss (1% price move = 5% on capital at 5x)
            if pnl_unrealized < -stop_loss_pct * leverage:
                exit_reason = f"stop_loss pnl={pnl_unrealized*100:.2f}%"
            # Trailing stop: once we've gained enough, trail from peak
            elif (peak_pnl >= trail_activation_pct * leverage
                  and pnl_unrealized < peak_pnl * trail_fraction):
                exit_reason = (f"trail_stop pnl={pnl_unrealized*100:.2f}% "
                               f"peak={peak_pnl*100:.2f}%")
            elif signals_held >= max_hold_signals:
                exit_reason = f"max_hold pnl={pnl_unrealized*100:.2f}%"
            # Fundamental exit: overshooting resolved AND profitable
            elif pnl_unrealized > 0 and s.d_P[1] < 1.0:
                exit_reason = f"resolved d_P_perp={s.d_P[1]:.2f} pnl={pnl_unrealized*100:.2f}%"

            if exit_reason:
                exit_price = s.prices[1]
                pnl_gross = direction * (exit_price - entry_price) / entry_price * leverage
                pnl_net = pnl_gross + tc_one_way * leverage
                trades.append(Trade(
                    entry_time=entry_time, exit_time=s.timestamp,
                    entry_spread=entry_price, exit_spread=exit_price,
                    direction=direction, entry_reason=entry_reason,
                    exit_reason=exit_reason,
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                ))
                equity.append(equity[-1] * (1 + pnl_net))
                timestamps.append(s.timestamp)
                in_position = False

        prev_spread = s.spread

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_threshold": dp_threshold,
                                   "stop_loss_pct": stop_loss_pct,
                                   "trail_activation_pct": trail_activation_pct,
                                   "trail_fraction": trail_fraction,
                                   "leverage": leverage,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


def backtest_p_avoid(
    signals: list,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
    name: str = "P_AVOID",
) -> PortfolioResult:
    """
    IPES Regime Avoidance: P2A with IPES stress filter.

    Same as P2A (cointegration mean-reversion) but SKIPS entry during
    IPES-detected stress:
      - Block entry when d^P_0 + d^P_1 > 2.0 (overshooting zone)
      - Block entry when E_i > threshold for either market

    Hypothesis: mean reversion works in calm markets but fails during
    cascades. IPES tells us WHEN to stay out. This is the opposite of P3
    (which enters during stress) — P_AVOID exits/avoids stress.

    This tests whether IPES improves P2A by REMOVING bad trades rather
    than adding good ones. This is a project hypothesis, not a paper claim.
    """
    tc = round_trip_cost(tx_cost_bps)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    direction = 0
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    signals_held = 0

    for s in signals:
        # Emergency: cointegration break
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        # IPES stress detection: exit if stress emerges while in position
        d_sum = s.d_P[0] + s.d_P[1]
        ipes_stress = (d_sum > 2.0) or (s.E[0] > E_threshold) or (s.E[1] > E_threshold)

        if in_position and ipes_stress:
            _close_position(direction, entry_spread, entry_time, entry_reason,
                            s, tc, trades, equity, timestamps,
                            f"ipes_stress d_sum={d_sum:.2f} E=[{s.E[0]:.2f},{s.E[1]:.2f}]",
                            entry_mid_price=entry_mid_price)
            in_position = False
            continue

        if not in_position:
            # Block entry during stress
            if ipes_stress:
                continue

            if abs(s.z_coint_resid) > z_entry:
                in_position = True
                direction = -1 if s.z_coint_resid > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"z_coint={s.z_coint_resid:.2f} d_sum={d_sum:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None
            spread_moved = abs(s.spread - entry_spread)

            if signals_held >= max_hold_signals:
                exit_reason = f"max_holding_period ({signals_held} signals)"
            elif abs(s.z_coint_resid) <= z_exit and spread_moved >= min_spread_exit:
                exit_reason = f"z-score reverted (ds={spread_moved:.2f})"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"z_entry": z_entry, "z_exit": z_exit,
                                   "E_threshold": E_threshold,
                                   "tx_cost_bps": tx_cost_bps,
                                   "min_spread_exit": min_spread_exit,
                                   "max_hold_signals": max_hold_signals})


# ── Always-on strategies: momentum and carry ────────────────────────────────


def backtest_p_mom(
    signals: list,
    ema_periods: int = 672,
    ema_band_pct: float = 0.005,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    leverage: int = 3,
    E_threshold_green: float = 0.3,
    E_threshold_red: float = 0.8,
    dp_red_threshold: float = 1.5,
    name: str = "P_MOM",
) -> PortfolioResult:
    """
    BTC Momentum with IPES Regime Filter.

    Always-on trend-following strategy:
      - Long BTC when price > EMA * (1 + band), flat when price < EMA * (1 - band)
      - Hysteresis band prevents whipsaw: must cross EMA by ema_band_pct to flip
      - IPES modulates position sizing:
        * Green (E_i < 0.3 both): full position
        * Yellow (0.3-0.8): half position (approximated as skip entry)
        * Red (E_i > 0.8 or d^P > 1.5): flatten immediately

    The red threshold is set high (E > 0.8, d^P > 1.5) because momentum
    strategies should ride through normal volatility — only flatten during
    genuine cascades. The old d^P > 1.0 threshold caused excessive churn.

    Uses perp for execution (leverage available).
    EMA computed from rolling signal prices.
    Default: 672 signals = 7 days at 15-min step (BTC momentum is multi-day).
    """
    tc_one_way = round_trip_cost(tx_cost_bps) / 2
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    # Build EMA from perp prices
    prices = [s.prices[1] for s in signals]
    alpha = 2.0 / (ema_periods + 1)

    in_position = False
    direction = 0
    entry_price = 0.0
    entry_time = None
    entry_reason = ""
    ema = prices[0] if prices else 0.0

    for i, s in enumerate(signals):
        # Update EMA
        ema = alpha * s.prices[1] + (1 - alpha) * ema
        if i < ema_periods:
            # Not enough data for EMA yet
            continue

        price = s.prices[1]

        # IPES regime assessment — high bar for red to avoid churn
        ipes_red = (s.E[0] > E_threshold_red or s.E[1] > E_threshold_red
                    or s.d_P[0] > dp_red_threshold or s.d_P[1] > dp_red_threshold)
        ipes_green = (s.E[0] < E_threshold_green and s.E[1] < E_threshold_green)

        # Momentum signal with hysteresis band
        # Enter long only when price crosses above EMA * (1 + band)
        # Exit long only when price crosses below EMA * (1 - band)
        ema_upper = ema * (1 + ema_band_pct)
        ema_lower = ema * (1 - ema_band_pct)

        if in_position:
            # Exit conditions
            exit_reason = None

            if ipes_red:
                exit_reason = f"ipes_red E=[{s.E[0]:.2f},{s.E[1]:.2f}]"
            elif price < ema_lower:
                exit_reason = f"ema_cross_below price={price:.0f} ema={ema:.0f} band={ema_lower:.0f}"

            if exit_reason:
                exit_price = s.prices[1]
                pnl_gross = direction * (exit_price - entry_price) / entry_price * leverage
                pnl_net = pnl_gross + tc_one_way * leverage
                trades.append(Trade(
                    entry_time=entry_time, exit_time=s.timestamp,
                    entry_spread=entry_price, exit_spread=exit_price,
                    direction=direction, entry_reason=entry_reason,
                    exit_reason=exit_reason,
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                ))
                equity.append(equity[-1] * (1 + pnl_net))
                timestamps.append(s.timestamp)
                in_position = False

        if not in_position and price > ema_upper and not ipes_red:
            # Enter long — require price above upper band (not just above EMA)
            if ipes_green or not s.cointegration_holds:
                # Green regime or no cointegration data: full entry
                in_position = True
                direction = 1
                entry_price = s.prices[1]
                entry_time = s.timestamp
                regime = "green" if ipes_green else "no_coint"
                entry_reason = f"ema_long price={price:.0f} ema={ema:.0f} regime={regime}"

                # Entry cost
                equity.append(equity[-1] * (1 + tc_one_way * leverage))
                timestamps.append(s.timestamp)

    # Close open position at end
    if in_position and signals:
        s = signals[-1]
        exit_price = s.prices[1]
        pnl_gross = direction * (exit_price - entry_price) / entry_price * leverage
        pnl_net = pnl_gross + tc_one_way * leverage
        trades.append(Trade(
            entry_time=entry_time, exit_time=s.timestamp,
            entry_spread=entry_price, exit_spread=exit_price,
            direction=direction, entry_reason=entry_reason,
            exit_reason="end_of_period",
            pnl_gross=pnl_gross, pnl_net=pnl_net,
        ))
        equity.append(equity[-1] * (1 + pnl_net))
        timestamps.append(s.timestamp)

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"ema_periods": ema_periods,
                                   "ema_band_pct": ema_band_pct,
                                   "leverage": leverage,
                                   "tx_cost_bps": tx_cost_bps,
                                   "E_threshold_green": E_threshold_green,
                                   "E_threshold_red": E_threshold_red,
                                   "dp_red_threshold": dp_red_threshold})


def backtest_p_carry(
    signals: list,
    funding_rate: float = 0.0001,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    E_threshold: float = 0.8,
    d_sum_threshold: float = 3.0,
    reentry_cooldown: int = 32,
    name: str = "P_CARRY",
) -> PortfolioResult:
    """
    Funding Rate Carry with IPES Shield.

    Always-on carry trade: long spot + short perp to earn funding rate.
    Funding is paid every 8 hours (~32 signals at 15-min step).

    IPES shield: exit only on genuine cascades, not normal market noise.
      - Exit: E_i > E_threshold (0.8) for either market OR d_sum > d_sum_threshold (3.0)
      - Re-enter: stress resolved AND cooldown elapsed (prevents whipsaw re-entry)

    The high thresholds mean the shield only triggers during real cascade events,
    not the frequent E_i > 0.5 noise that caused 2,415 trades with the old params.
    The cooldown prevents immediately re-entering after an exit, reducing TC drag.

    funding_rate: average per-settlement rate (default 0.0001 = 1 bp per 8h).
    Actual rates vary; this is a conservative estimate for 2024.
    """
    tc = round_trip_cost(tx_cost_bps)
    funding_per_signal = funding_rate / 32  # ~32 signals per 8-hour period (15-min step)
    trades = []
    equity = [1.0]
    timestamps = [signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")]

    in_position = False
    entry_spread = 0.0
    entry_time = None
    entry_reason = ""
    entry_mid_price = 0.0
    funding_earned = 0.0
    signals_held = 0
    cooldown_remaining = 0  # signals to wait before re-entering

    for s in signals:
        d_sum = s.d_P[0] + s.d_P[1]
        ipes_stress = (s.E[0] > E_threshold or s.E[1] > E_threshold
                       or d_sum > d_sum_threshold)

        if in_position:
            signals_held += 1
            # Accrue funding (long spot + short perp earns funding when rate > 0)
            funding_earned += funding_per_signal

            if ipes_stress:
                # Shield: exit to protect carry gains
                spread_pnl = -(s.spread - entry_spread) / entry_mid_price  # short perp
                pnl_gross = spread_pnl + funding_earned
                pnl_net = pnl_gross + tc
                trades.append(Trade(
                    entry_time=entry_time, exit_time=s.timestamp,
                    entry_spread=entry_spread, exit_spread=s.spread,
                    direction=-1, entry_reason=entry_reason,
                    exit_reason=f"ipes_shield d_sum={d_sum:.2f} E=[{s.E[0]:.2f},{s.E[1]:.2f}] "
                               f"funding={funding_earned*10000:.1f}bps held={signals_held}sigs",
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                ))
                equity.append(equity[-1] * (1 + pnl_net))
                timestamps.append(s.timestamp)
                in_position = False
                funding_earned = 0.0
                cooldown_remaining = reentry_cooldown

        if not in_position:
            # Tick down cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            if not ipes_stress:
                # Enter carry: long spot + short perp
                in_position = True
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                entry_reason = f"carry d_sum={d_sum:.2f} E=[{s.E[0]:.2f},{s.E[1]:.2f}]"
                funding_earned = 0.0
                signals_held = 0

    # Close open position at end
    if in_position and signals:
        s = signals[-1]
        spread_pnl = -(s.spread - entry_spread) / entry_mid_price
        pnl_gross = spread_pnl + funding_earned
        pnl_net = pnl_gross + tc
        trades.append(Trade(
            entry_time=entry_time, exit_time=s.timestamp,
            entry_spread=entry_spread, exit_spread=s.spread,
            direction=-1, entry_reason=entry_reason,
            exit_reason=f"end_of_period funding={funding_earned*10000:.1f}bps",
            pnl_gross=pnl_gross, pnl_net=pnl_net,
        ))
        equity.append(equity[-1] * (1 + pnl_net))
        timestamps.append(s.timestamp)

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"funding_rate": funding_rate,
                                   "tx_cost_bps": tx_cost_bps,
                                   "E_threshold": E_threshold,
                                   "d_sum_threshold": d_sum_threshold,
                                   "reentry_cooldown": reentry_cooldown})


# ── Run all portfolios ───────────────────────────────────────────────────────


def run_all_portfolios(
    signals: list,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
) -> list:
    """Run all portfolio strategies on the same signal stream.

    Benchmarks:
      P0    -- Never trade (null benchmark for spread strategies)
      P_BH  -- Buy & hold BTC (directional benchmark)

    Spread strategies:
      P2A     -- Standard cointegration mean-reversion
      P3_IPES -- Cointegration + IPES filter (enter during stress)
      P_AVOID -- Cointegration + IPES filter (avoid stress)

    Directional strategy:
      P_DIR -- Directional cascade fade using IPES overshooting detection
    """
    results = []

    # ── Benchmarks ──────────────────────────────────────────────────
    # P0 -- Never trade (null benchmark)
    eq = pd.Series([1.0], index=[signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")])
    results.append(PortfolioResult(name="P0_NeverTrade", trades=[], equity_curve=eq, params={}))

    # P_BH -- Buy & hold BTC (directional benchmark)
    results.append(backtest_p_bh(signals))

    # ── Spread strategies ───────────────────────────────────────────
    # P2A -- Standard cointegration mean-reversion
    results.append(backtest_p2(signals, has_stop_loss=False,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit))

    # P3_IPES -- Enter during IPES-detected overshooting
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit,
                               z_entry=1.0,
                               name="P3_IPES"))

    # P_AVOID -- P2A but skip stress periods (opposite of P3)
    results.append(backtest_p_avoid(signals, E_threshold=E_threshold,
                                    tx_cost_bps=tx_cost_bps,
                                    min_spread_exit=min_spread_exit))

    # ── Directional strategy ────────────────────────────────────────
    # P_DIR -- Directional cascade fade (unhedged, uses perp price moves)
    results.append(backtest_p_dir(signals, tx_cost_bps=tx_cost_bps))

    # ── Always-on strategies ───────────────────────────────────────
    # P_MOM -- Momentum with IPES regime filter
    results.append(backtest_p_mom(signals, tx_cost_bps=tx_cost_bps))

    # P_CARRY -- Funding rate carry with IPES shield
    # Uses its own higher E_threshold (0.8) and d_sum_threshold (3.0) defaults
    # to avoid the excessive round trips that plagued the old E=0.5 config
    results.append(backtest_p_carry(signals, tx_cost_bps=tx_cost_bps))

    return results


def results_table(results: list) -> pd.DataFrame:
    """Compile metrics from all portfolio results into a comparison table."""
    rows = [r.metrics() for r in results]
    df = pd.DataFrame(rows)
    df = df.set_index("portfolio")
    return df


# ── Robustness sweep ─────────────────────────────────────────────────────────


def robustness_sweep(
    spot_close: pd.Series,
    perp_close: pd.Series,
    interval: str = "1s",
    E_thresholds: Optional[list] = None,
    W_values: Optional[list] = None,
    K_values: Optional[list] = None,
    tx_cost_grid: Optional[list] = None,
    deterministic_values: Optional[list] = None,
    beta_constrained_values: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep over parameter grid, running P3 for each combination.

    Returns a DataFrame with one row per parameter combination.
    """
    if E_thresholds is None:
        E_thresholds = [0.3, 0.5, 0.75, 1.0, 1.5]
    if W_values is None:
        W_values = [1800, 3600, 7200]  # 30min, 60min, 2hr windows for 1s data
    if K_values is None:
        K_values = [3, 5, 7]
    if tx_cost_grid is None:
        tx_cost_grid = [0, 2, 5, 10, 15]
    if deterministic_values is None:
        deterministic_values = ["ci", "co"]
    if beta_constrained_values is None:
        beta_constrained_values = [True, False]

    # Inner-join and prep data once
    raw = pd.concat([spot_close, perp_close], axis=1, join="inner").dropna()
    raw_prices = raw.copy()
    raw_prices.columns = ["BTC_SPOT", "BTC_PERP"]

    # Log prices -- simple log, no cross-rate
    log_p = np.log(raw_prices)
    log_p.columns = ["BTC_SPOT", "BTC_PERP"]

    all_rows = []
    estimation_combos = list(itertools.product(
        W_values, K_values, deterministic_values, beta_constrained_values
    ))

    for W, K, det, beta_c in estimation_combos:
        if verbose:
            print(f"\nEstimating: W={W}, K={K}, det={det}, "
                  f"beta_constrained={beta_c}")

        try:
            result = rolling_estimate(
                log_p, raw_prices, interval=interval,
                W=W, K=K, deterministic=det,
                constrain_beta=beta_c, verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            continue

        if not result.signals:
            if verbose:
                print("  No signals produced")
            continue

        # Sweep over E_threshold and tx_cost
        for E_thresh, tx_bps in itertools.product(E_thresholds, tx_cost_grid):
            p3 = backtest_p3(
                result.signals,
                E_threshold=E_thresh,
                tx_cost_bps=tx_bps,
                min_spread_exit=DEFAULT_MIN_SPREAD_EXIT,
            )
            m = p3.metrics()
            m.update({
                "W": W, "K": K, "deterministic": det,
                "beta_constrained": beta_c,
                "E_threshold": E_thresh, "tx_cost_bps": tx_bps,
            })
            all_rows.append(m)

    df = pd.DataFrame(all_rows)
    if verbose and len(df) > 0:
        print(f"\nSweep complete: {len(df)} configurations tested")
    return df


# ── Strategy-level parameter sweeps ──────────────────────────────────────────
#
# These sweep strategy parameters on a pre-computed signal stream (fast, O(n)
# per config). Unlike robustness_sweep which re-estimates VECM for each combo,
# these only vary the backtest-level params.


def p_mom_sweep(
    signals: list,
    ema_periods_values: Optional[list] = None,
    E_threshold_red_values: Optional[list] = None,
    dp_red_threshold_values: Optional[list] = None,
    ema_band_pct_values: Optional[list] = None,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    leverage: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep P_MOM parameters over a grid. Returns one row per config."""
    if ema_periods_values is None:
        ema_periods_values = [96, 192, 336, 672, 1344]
    if E_threshold_red_values is None:
        E_threshold_red_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if dp_red_threshold_values is None:
        dp_red_threshold_values = [0.8, 1.0, 1.2, 1.5, 2.0]
    if ema_band_pct_values is None:
        ema_band_pct_values = [0.0, 0.003, 0.005, 0.01, 0.02]

    rows = []
    combos = list(itertools.product(
        ema_periods_values, E_threshold_red_values,
        dp_red_threshold_values, ema_band_pct_values,
    ))
    if verbose:
        print(f"  P_MOM sweep: {len(combos)} configurations")

    for ema_p, e_red, dp_red, band in combos:
        res = backtest_p_mom(
            signals, ema_periods=ema_p, E_threshold_red=e_red,
            dp_red_threshold=dp_red, ema_band_pct=band,
            tx_cost_bps=tx_cost_bps, leverage=leverage,
        )
        m = res.metrics()
        m.update({"ema_periods": ema_p, "E_threshold_red": e_red,
                  "dp_red_threshold": dp_red, "ema_band_pct": band})
        rows.append(m)

    df = pd.DataFrame(rows)
    if verbose:
        viable = (df["sharpe"] > 0).sum()
        print(f"  P_MOM sweep done: {viable}/{len(df)} configs with Sharpe > 0")
    return df


def p_carry_sweep(
    signals: list,
    E_threshold_values: Optional[list] = None,
    d_sum_threshold_values: Optional[list] = None,
    reentry_cooldown_values: Optional[list] = None,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep P_CARRY parameters over a grid. Returns one row per config."""
    if E_threshold_values is None:
        E_threshold_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if d_sum_threshold_values is None:
        d_sum_threshold_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    if reentry_cooldown_values is None:
        reentry_cooldown_values = [0, 16, 32, 64]

    rows = []
    combos = list(itertools.product(
        E_threshold_values, d_sum_threshold_values, reentry_cooldown_values,
    ))
    if verbose:
        print(f"  P_CARRY sweep: {len(combos)} configurations")

    for e_th, d_sum, cooldown in combos:
        res = backtest_p_carry(
            signals, E_threshold=e_th, d_sum_threshold=d_sum,
            reentry_cooldown=cooldown, tx_cost_bps=tx_cost_bps,
        )
        m = res.metrics()
        m.update({"E_threshold": e_th, "d_sum_threshold": d_sum,
                  "reentry_cooldown": cooldown})
        rows.append(m)

    df = pd.DataFrame(rows)
    if verbose:
        viable = (df["sharpe"] > 0).sum()
        print(f"  P_CARRY sweep done: {viable}/{len(df)} configs with Sharpe > 0")
    return df


def p_dir_sweep(
    signals: list,
    dp_threshold_values: Optional[list] = None,
    stop_loss_pct_values: Optional[list] = None,
    trail_activation_pct_values: Optional[list] = None,
    max_hold_signals_values: Optional[list] = None,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    leverage: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep P_DIR parameters over a grid. Returns one row per config."""
    if dp_threshold_values is None:
        dp_threshold_values = [0.8, 1.0, 1.2, 1.5, 2.0]
    if stop_loss_pct_values is None:
        stop_loss_pct_values = [0.005, 0.01, 0.02, 0.03]
    if trail_activation_pct_values is None:
        trail_activation_pct_values = [0.005, 0.01, 0.02]
    if max_hold_signals_values is None:
        max_hold_signals_values = [144, 288, 576]

    rows = []
    combos = list(itertools.product(
        dp_threshold_values, stop_loss_pct_values,
        trail_activation_pct_values, max_hold_signals_values,
    ))
    if verbose:
        print(f"  P_DIR sweep: {len(combos)} configurations")

    for dp_th, sl, trail, mh in combos:
        res = backtest_p_dir(
            signals, dp_threshold=dp_th, stop_loss_pct=sl,
            trail_activation_pct=trail, max_hold_signals=mh,
            tx_cost_bps=tx_cost_bps, leverage=leverage,
        )
        m = res.metrics()
        m.update({"dp_threshold": dp_th, "stop_loss_pct": sl,
                  "trail_activation_pct": trail, "max_hold_signals": mh})
        rows.append(m)

    df = pd.DataFrame(rows)
    if verbose:
        viable = (df["sharpe"] > 0).sum()
        print(f"  P_DIR sweep done: {viable}/{len(df)} configs with Sharpe > 0")
    return df


# ── Full pipeline: data -> signals -> backtest -> metrics ────────────────────


def run_full_pipeline(
    spot_close: pd.Series,
    perp_close: pd.Series,
    interval: str = "1s",
    W: Optional[int] = None,
    K: int = 5,
    deterministic: str = "ci",
    constrain_beta: bool = True,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    verbose: bool = True,
) -> tuple:
    """
    End-to-end: estimate -> backtest all portfolios -> metrics.

    Returns (RollingResult, list[PortfolioResult], metrics_DataFrame).
    """
    if verbose:
        print(f"Running full pipeline: BTC_SPOT vs BTC_PERP")
        print(f"  {len(spot_close)} obs, interval={interval}")

    rolling = run_spot_perp_estimation(
        spot_close, perp_close, interval=interval,
        W=W, K=K, deterministic=deterministic,
        constrain_beta=constrain_beta, verbose=verbose,
    )

    if not rolling.signals:
        print("  No signals produced -- insufficient data or all windows failed")
        return rolling, [], pd.DataFrame()

    portfolios = run_all_portfolios(
        rolling.signals,
        E_threshold=E_threshold,
        tx_cost_bps=tx_cost_bps,
        min_spread_exit=min_spread_exit,
    )

    metrics = results_table(portfolios)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Results: BTC_SPOT vs BTC_PERP | "
              f"{interval} | W={rolling.params['W']} K={K}")
        print(f"{'=' * 70}")
        print(metrics.to_string())
        print()

    return rolling, portfolios, metrics


# ── Bootstrap confidence intervals ───────────────────────────────────────────


def bootstrap_drawdown_ci(
    signals: list,
    portfolio_a_fn,
    portfolio_b_fn,
    n_bootstrap: int = 1000,
    block_size: int = 50,
    seed: int = 42,
) -> dict:
    """
    Block bootstrap test for max drawdown difference between two portfolios.

    Uses a circular block bootstrap on the signal stream to preserve temporal
    structure. For each iteration, resamples contiguous blocks and re-runs
    both portfolios, recording the max drawdown difference.

    Returns dict with: observed_diff, mean_diff, ci_lower, ci_upper,
                       p_value, n_bootstrap, a_name, b_name.
    """
    rng = np.random.RandomState(seed)
    n = len(signals)
    if n < block_size:
        block_size = max(n // 4, 2)

    # Observed values
    res_a = portfolio_a_fn(signals)
    res_b = portfolio_b_fn(signals)
    dd_a = res_a.metrics()["max_drawdown"]
    dd_b = res_b.metrics()["max_drawdown"]
    observed_diff = dd_a - dd_b  # DD values are negative; diff < 0 means B better

    # Bootstrap
    diffs = []
    n_blocks = (n + block_size - 1) // block_size

    for _ in range(n_bootstrap):
        # Circular block bootstrap: pick random start indices
        starts = rng.randint(0, n, size=n_blocks)
        boot_signals = []
        for s_idx in starts:
            for j in range(block_size):
                idx = (s_idx + j) % n
                boot_signals.append(signals[idx])
            if len(boot_signals) >= n:
                break
        boot_signals = boot_signals[:n]

        try:
            ba = portfolio_a_fn(boot_signals)
            bb = portfolio_b_fn(boot_signals)
            d_a = ba.metrics()["max_drawdown"]
            d_b = bb.metrics()["max_drawdown"]
            diffs.append(d_a - d_b)
        except Exception:
            continue

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {
            "observed_diff": observed_diff,
            "mean_diff": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "p_value": np.nan,
            "n_bootstrap": 0,
            "a_name": res_a.name, "b_name": res_b.name,
        }

    return {
        "observed_diff": observed_diff,
        "mean_diff": np.mean(diffs),
        "ci_lower": np.percentile(diffs, 2.5),
        "ci_upper": np.percentile(diffs, 97.5),
        "p_value": np.mean(diffs >= 0),
        "n_bootstrap": len(diffs),
        "a_name": res_a.name, "b_name": res_b.name,
    }


def bootstrap_pnl_ci(
    signals: list,
    portfolio_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI on average trade PnL for a single portfolio.

    Resamples completed trades with replacement (standard i.i.d. bootstrap
    on the trade-level PnLs).

    Returns dict with: portfolio, observed_mean, ci_lower, ci_upper, n_trades.
    """
    rng = np.random.RandomState(seed)
    res = portfolio_fn(signals)
    pnls = np.array([t.pnl_net for t in res.trades])

    if len(pnls) < 2:
        return {
            "portfolio": res.name,
            "observed_mean": np.mean(pnls) if len(pnls) > 0 else 0.0,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_trades": len(pnls),
        }

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    return {
        "portfolio": res.name,
        "observed_mean": np.mean(pnls),
        "ci_lower": np.percentile(boot_means, 2.5),
        "ci_upper": np.percentile(boot_means, 97.5),
        "n_trades": len(pnls),
    }


def run_bootstrap_tests(
    signals: list,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Run bootstrap CI tests comparing P3_IPES vs P2A on drawdown and PnL.

    Returns dict with keys: 'drawdown_tests', 'pnl_tests'.
    """
    def make_p2a(sigs):
        return backtest_p2(sigs, has_stop_loss=False, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit)

    def make_p3(sigs):
        return backtest_p3(sigs, E_threshold=E_threshold, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit, z_entry=1.0,
                          name="P3_IPES")

    drawdown_tests = []
    drawdown_tests.append(bootstrap_drawdown_ci(signals, make_p2a, make_p3,
                                                n_bootstrap=n_bootstrap))

    pnl_tests = []
    for fn in [make_p2a, make_p3]:
        pnl_tests.append(bootstrap_pnl_ci(signals, fn, n_bootstrap=n_bootstrap))

    return {"drawdown_tests": drawdown_tests, "pnl_tests": pnl_tests}


# ── LowZ threshold sweep ────────────────────────────────────────────────────


def lowz_threshold_sweep(
    signals: list,
    z_entry_values: list | None = None,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
) -> pd.DataFrame:
    """
    Sweep z_entry threshold for P3, keeping all other filters at defaults.

    Returns DataFrame with one row per z_entry value.
    """
    if z_entry_values is None:
        z_entry_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    rows = []
    for z in z_entry_values:
        res = backtest_p3(
            signals, E_threshold=E_threshold, z_entry=z,
            tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
            name=f"P3_z{z:.2f}",
        )
        m = res.metrics()
        m["z_entry"] = z
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["z_entry", "portfolio", "n_trades", "total_return", "ann_return",
            "sharpe", "max_drawdown", "calmar", "avg_trade_pnl", "win_rate"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.load_data import load_aligned_pair

    print("Loading data...")
    spot_c, perp_c = load_aligned_pair()

    # Use first week of data for testing
    start = spot_c.index[0]
    end = start + pd.Timedelta(days=7)
    spot_slice = spot_c[start:end]
    perp_slice = perp_c[start:end]
    print(f"Test period: {spot_slice.index[0]} to {spot_slice.index[-1]} ({len(spot_slice)} rows)")

    rolling, portfolios, metrics = run_full_pipeline(
        spot_slice, perp_slice, interval="1m", W=360, K=5, E_threshold=0.5,
        tx_cost_bps=5, verbose=True,
    )

    # Show trades from P3
    if portfolios:
        p3 = [p for p in portfolios if p.name == "P3_Full"][0]
        print(f"\nP3 trades: {p3.n_trades}")
        for t in p3.trades[:5]:
            print(f"  {t.entry_time} -> {t.exit_time}: "
                  f"dir={t.direction:+d} pnl={t.pnl_net:+.6f} [{t.exit_reason}]")
