"""
Walk-forward backtesting engine for all four portfolios.

Consumes signals from vecm_ipes.rolling_estimate() and applies the
decision logic adapted for BTC spot vs BTC perpetual futures.

Portfolios:
    P1 -- Distance Method (benchmark)
    P2 -- Standard Cointegration (benchmark)
    P3 -- Cointegration + IPES Filter (core contribution)
    P4 -- Cointegration + Standard Stop-Loss

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
DEFAULT_TX_COST_BPS = 5        # ~3 bps futures + ~2 bps spot at maker rates
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
                "win_rate": 0.0,
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
        avg_pnl = np.mean(pnls) if pnls else 0.0
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0

        return {
            "portfolio": self.name,
            "n_trades": len(self.trades),
            "total_return": total_ret,
            "ann_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "avg_trade_pnl": avg_pnl,
            "win_rate": win_rate,
        }


# ── Transaction cost ─────────────────────────────────────────────────────────

def round_trip_cost(cost_bps: float) -> float:
    """
    Round-trip transaction cost as a log-return drag.
    Formula: 2 * ln((1-C)/(1+C)) where C = cost_bps / 10000
    """
    C = cost_bps / 10_000
    if C >= 1.0:
        return -np.inf
    return 2.0 * log((1 - C) / (1 + C))


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
                     entry_mid_price=None):
    """Helper to close a position and record the trade.

    PnL is normalized by entry_mid_price to produce a return (not a dollar amount).
    For BTC at ~$60K, spread moves of $10 should produce ~0.016% returns, not 1000% returns.
    """
    pnl_gross_raw = direction * (s.spread - entry_spread)
    # Normalize to return: divide by entry midpoint price
    if entry_mid_price and entry_mid_price > 0:
        pnl_gross = pnl_gross_raw / entry_mid_price
    else:
        pnl_gross = pnl_gross_raw
    pnl_net = pnl_gross + tc
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


def backtest_p5(
    signals: list,
    dp_threshold: float = 1.2,
    z_trans_min: float = 0.5,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = 48,
    name: str = "P5_OvershootFade",
) -> PortfolioResult:
    """
    Portfolio 5 -- Overshooting Fade.

    Directly exploits the paper's core finding: when one market overshoots
    (d^P > threshold), it has overreacted and will revert. Trade against
    the overshoot.

    Entry:
      - Cointegration holds
      - d^P[overshooting_leg] > dp_threshold (confirmed overshooting)
      - |z_trans| > z_trans_min (transitory dislocation exists)
    Exit:
      - d^P[overshooting_leg] < 1.0 (overshooting resolved)
      - OR z_trans reverted AND spread moved
      - OR max hold exceeded
      - OR cointegration break
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

    for s in signals:
        if not s.cointegration_holds:
            if in_position:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, "cointegration_break",
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        if not in_position:
            # Find which market is overshooting
            overshoot_leg = -1
            if s.d_P[0] > dp_threshold and s.d_P[1] > dp_threshold:
                overshoot_leg = 0 if s.d_P[0] > s.d_P[1] else 1
            elif s.d_P[0] > dp_threshold:
                overshoot_leg = 0
            elif s.d_P[1] > dp_threshold:
                overshoot_leg = 1

            if overshoot_leg >= 0 and abs(s.z_trans) > z_trans_min:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = (f"overshoot d_P[{overshoot_leg}]={s.d_P[overshoot_leg]:.2f} "
                               f"z_T={s.z_trans:.2f}")
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                entry_overshoot_leg = overshoot_leg
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None

            if signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} signals)"
            elif s.d_P[entry_overshoot_leg] < 1.0:
                exit_reason = f"overshoot_resolved d_P={s.d_P[entry_overshoot_leg]:.2f}"
            elif abs(s.z_trans) <= 0.5:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted (ds={spread_moved:.2f})"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_threshold": dp_threshold,
                                   "z_trans_min": z_trans_min,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


def backtest_p6(
    signals: list,
    dp_sum_entry: float = 1.8,
    dp_sum_exit: float = 1.5,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = 36,
    name: str = "P6_ConflictFade",
) -> PortfolioResult:
    """
    Portfolio 6 -- Conflict Zone Fade.

    When d^P_0 + d^P_1 approaches or exceeds 2 (overshooting zone from
    the paper), both markets have collectively overreacted. Fade the move.

    Entry:
      - Cointegration holds
      - d^P_0 + d^P_1 > dp_sum_entry (approaching/in overshooting zone)
    Exit:
      - d^P_0 + d^P_1 < dp_sum_exit (back in reliable zone)
      - OR max hold exceeded
      - OR cointegration break
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
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        d_sum = s.d_P[0] + s.d_P[1]

        if not in_position:
            if d_sum > dp_sum_entry:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"conflict_zone d_sum={d_sum:.2f} z_T={s.z_trans:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None

            if signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} signals)"
            elif d_sum < dp_sum_exit:
                exit_reason = f"conflict_resolved d_sum={d_sum:.2f}"
            elif abs(s.z_trans) <= 0.3:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted (ds={spread_moved:.2f})"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_sum_entry": dp_sum_entry,
                                   "dp_sum_exit": dp_sum_exit,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


def backtest_p7(
    signals: list,
    z_entry: float = 1.5,
    z_exit: float = 0.3,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = 0.5,
    max_hold_signals: int = 12,
    name: str = "P7_TransScalp",
) -> PortfolioResult:
    """
    Portfolio 7 -- Transitory Shock Scalper.

    Quick in-and-out on transitory shocks. Only requires cointegration.
    Designed for 1s data's high frequency — tight exits, short holds.

    Entry: |z_trans| > z_entry AND cointegration holds
    Exit: |z_trans| < z_exit OR max hold
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
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        if not in_position:
            if abs(s.z_trans) > z_entry:
                in_position = True
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = f"z_trans={s.z_trans:.2f}"
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None

            if signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} signals)"
            elif abs(s.z_trans) <= z_exit:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted={s.z_trans:.2f} (ds={spread_moved:.2f})"
                else:
                    exit_reason = f"z_trans_reverted={s.z_trans:.2f}"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"z_entry": z_entry, "z_exit": z_exit,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


def backtest_p8(
    signals: list,
    E_low: float = 0.3,
    E_high: float = 0.7,
    z_entry_efficient: float = 2.0,
    z_entry_mid: float = 1.0,
    z_entry_stressed: float = 0.5,
    z_exit: float = DEFAULT_Z_EXIT,
    z_perm_max: float = DEFAULT_Z_PERM_MAX,
    z_perm_spike: float = DEFAULT_Z_PERM_SPIKE,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = DEFAULT_MAX_HOLD_SIGNALS,
    name: str = "P8_IPESAdaptive",
) -> PortfolioResult:
    """
    Portfolio 8 -- IPES Adaptive Entry.

    Uses all P3 entry gates but adapts z_entry based on current pricing
    error regime. Higher E_i = more aggressive entry (lower z threshold).

    Regime classification based on max(E_0, E_1):
      - Efficient (E < E_low):  z_entry = z_entry_efficient (conservative)
      - Mid (E_low <= E < E_high): z_entry = z_entry_mid
      - Stressed (E >= E_high): z_entry = z_entry_stressed (aggressive)
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
    prev_E = None
    signals_held = 0

    for s in signals:
        # Emergency exits
        emergency_exit = None
        if not s.cointegration_holds:
            emergency_exit = "cointegration_break"
        elif not s.alpha_signs_opposite:
            emergency_exit = "alpha_diverging"
        elif s.d_P[0] < 0 or s.d_P[1] < 0:
            emergency_exit = f"perverse_response d_P=[{s.d_P[0]:.2f},{s.d_P[1]:.2f}]"

        if emergency_exit and in_position:
            _close_position(direction, entry_spread, entry_time, entry_reason,
                            s, tc, trades, equity, timestamps, emergency_exit,
                            entry_mid_price=entry_mid_price)
            in_position = False

        if emergency_exit:
            prev_E = s.E.copy()
            continue

        # Adaptive z_entry based on pricing error regime
        max_E = max(s.E[0], s.E[1])
        if max_E >= E_high:
            adaptive_z = z_entry_stressed
        elif max_E >= E_low:
            adaptive_z = z_entry_mid
        else:
            adaptive_z = z_entry_efficient

        dislo = _identify_dislocation_leg(s.prices)

        if not in_position:
            # Bilateral block
            if s.E[0] > 0.5 and s.E[1] > 0.5:
                prev_E = s.E.copy()
                continue

            if s.E[dislo] <= 0.3:  # minimal pricing error — skip
                prev_E = s.E.copy()
                continue

            if s.d_P[dislo] <= 1.0:
                prev_E = s.E.copy()
                continue

            if abs(s.z_trans) <= adaptive_z:
                prev_E = s.E.copy()
                continue

            if abs(s.z_perm) >= z_perm_max:
                prev_E = s.E.copy()
                continue

            in_position = True
            direction = -1 if s.z_trans > 0 else 1
            entry_spread = s.spread
            entry_time = s.timestamp
            entry_reason = (f"adaptive z={adaptive_z:.1f} E_max={max_E:.2f} "
                           f"d_P={s.d_P[dislo]:.2f} z_T={s.z_trans:.2f}")
            entry_mid_price = (s.prices[0] + s.prices[1]) / 2
            signals_held = 0
        else:
            signals_held += 1
            exit_reason = None

            if signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} signals)"
            elif abs(s.z_perm) >= z_perm_spike:
                exit_reason = f"perm_spike z_P={s.z_perm:.2f}"
            elif (prev_E is not None and
                  prev_E[dislo] > 0.5 and s.E[dislo] <= 0.5):
                exit_reason = "E_i_regime_flip"
            elif abs(s.z_trans) <= z_exit:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted={s.z_trans:.2f}"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

        prev_E = s.E.copy()

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"E_low": E_low, "E_high": E_high,
                                   "z_entry_efficient": z_entry_efficient,
                                   "z_entry_mid": z_entry_mid,
                                   "z_entry_stressed": z_entry_stressed,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


def backtest_p9(
    signals: list,
    dp_perp_min: float = 1.3,
    E_perp_min: float = 0.3,
    E_perp_exit: float = 0.15,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
    max_hold_signals: int = 48,
    name: str = "P9_CascadeFade",
) -> PortfolioResult:
    """
    Portfolio 9 -- Cascade Fade.

    Specifically targets liquidation cascade patterns where the futures
    market overreacts due to forced liquidations. Enters when perp shows
    strong overshooting with high pricing error. Exits when pricing
    error normalizes.

    Entry:
      - Cointegration holds
      - d^P[1] (perp) > dp_perp_min (perp overshooting)
      - E[1] (perp) > E_perp_min (perp has significant pricing error)
    Exit:
      - E[1] < E_perp_exit (perp pricing error normalized)
      - OR d^P[1] < 1.0 (perp overshooting resolved)
      - OR max hold exceeded
      - OR cointegration break
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
                                entry_mid_price=entry_mid_price)
                in_position = False
            continue

        if not in_position:
            # Perp (index 1) must be overshooting with high pricing error
            if s.d_P[1] > dp_perp_min and s.E[1] > E_perp_min:
                in_position = True
                # Fade the perp overshoot: perp overshot → short perp, long spot
                direction = -1 if s.z_trans > 0 else 1
                entry_spread = s.spread
                entry_time = s.timestamp
                entry_reason = (f"cascade d_P_perp={s.d_P[1]:.2f} "
                               f"E_perp={s.E[1]:.3f} z_T={s.z_trans:.2f}")
                entry_mid_price = (s.prices[0] + s.prices[1]) / 2
                signals_held = 0
        else:
            signals_held += 1
            exit_reason = None

            if signals_held >= max_hold_signals:
                exit_reason = f"max_hold ({signals_held} signals)"
            elif s.E[1] < E_perp_exit:
                exit_reason = f"perp_error_normalized E={s.E[1]:.3f}"
            elif s.d_P[1] < 1.0:
                exit_reason = f"perp_overshoot_resolved d_P={s.d_P[1]:.2f}"
            elif abs(s.z_trans) <= 0.3:
                spread_moved = abs(s.spread - entry_spread)
                if spread_moved >= min_spread_exit:
                    exit_reason = f"z_trans_reverted (ds={spread_moved:.2f})"

            if exit_reason:
                _close_position(direction, entry_spread, entry_time, entry_reason,
                                s, tc, trades, equity, timestamps, exit_reason,
                                entry_mid_price=entry_mid_price)
                in_position = False

    eq_series = pd.Series(equity, index=timestamps[:len(equity)])
    return PortfolioResult(name=name, trades=trades, equity_curve=eq_series,
                           params={"dp_perp_min": dp_perp_min,
                                   "E_perp_min": E_perp_min,
                                   "E_perp_exit": E_perp_exit,
                                   "tx_cost_bps": tx_cost_bps,
                                   "max_hold_signals": max_hold_signals})


# ── Run all portfolios ───────────────────────────────────────────────────────


def run_all_portfolios(
    signals: list,
    E_threshold: float = DEFAULT_E_THRESHOLD,
    tx_cost_bps: float = DEFAULT_TX_COST_BPS,
    min_spread_exit: float = DEFAULT_MIN_SPREAD_EXIT,
) -> list:
    """Run all portfolio variants on the same signal stream.

    Includes benchmark strategies (P1-P2), IPES-filtered (P3), stop-loss (P4),
    and new price-discovery strategies (P5-P9) that exploit the paper's
    insights about overshooting and conflict zones.
    """
    results = []

    # P0 -- Never trade (null benchmark)
    eq = pd.Series([1.0], index=[signals[0].timestamp if signals else pd.Timestamp.now(tz="UTC")])
    results.append(PortfolioResult(name="P0_NeverTrade", trades=[], equity_curve=eq, params={}))

    # ── Benchmarks ──────────────────────────────────────────────────
    # P1A, P1B (distance method)
    results.append(backtest_p1(signals, has_stop_loss=False,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit))
    results.append(backtest_p1(signals, has_stop_loss=True,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit))

    # P2A, P2B (cointegration)
    results.append(backtest_p2(signals, has_stop_loss=False,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit))
    results.append(backtest_p2(signals, has_stop_loss=True,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit))

    # ── P3: IPES Filter (core) ──────────────────────────────────────
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit,
                               name="P3_Full"))

    # P3 with relaxed z_trans entry threshold
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               z_entry=1.0, name="P3_LowZ1"))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               z_entry=0.5, name="P3_LowZ05"))

    # P3 hold period variants (for 1s data with 5-min step)
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               max_hold_signals=12, name="P3_Hold1h"))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               max_hold_signals=48, name="P3_Hold4h"))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               max_hold_signals=144, name="P3_Hold12h"))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               max_hold_signals=288, name="P3_Hold24h"))

    # P3 ablations (which filter matters most?)
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P3_NoBilateral", use_bilateral_block=False))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P3_NoEiThresh", use_ei_threshold=False))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P3_NoDpOvershoot", use_dp_overshoot=False))
    results.append(backtest_p3(signals, E_threshold=E_threshold,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P3_AllFiltersOff",
                               use_alpha_diverging_block=False,
                               use_z_perm_max_block=False,
                               use_z_trans_entry=False,
                               use_perverse_block=False,
                               use_bilateral_block=False,
                               use_ei_threshold=False,
                               use_dp_overshoot=False))

    # ── P4: Stop-loss variants ──────────────────────────────────────
    for sl in P4_STOP_LEVELS_PCT:
        results.append(backtest_p4(signals, stop_loss_pct=sl,
                                   tx_cost_bps=tx_cost_bps,
                                   min_spread_exit=min_spread_exit))

    # ── P5: Overshooting Fade ───────────────────────────────────────
    results.append(backtest_p5(signals, dp_threshold=1.2, z_trans_min=0.5,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P5_OvershootFade"))
    results.append(backtest_p5(signals, dp_threshold=1.1, z_trans_min=0.3,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P5_Aggressive"))
    results.append(backtest_p5(signals, dp_threshold=1.3, z_trans_min=1.0,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P5_Conservative"))

    # ── P6: Conflict Zone Fade ──────────────────────────────────────
    results.append(backtest_p6(signals, dp_sum_entry=1.8, dp_sum_exit=1.5,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P6_ConflictFade"))
    results.append(backtest_p6(signals, dp_sum_entry=1.6, dp_sum_exit=1.3,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P6_Aggressive"))
    results.append(backtest_p6(signals, dp_sum_entry=2.0, dp_sum_exit=1.5,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P6_StrictZone"))

    # ── P7: Transitory Scalper ──────────────────────────────────────
    results.append(backtest_p7(signals, z_entry=1.5, z_exit=0.3,
                               tx_cost_bps=tx_cost_bps,
                               name="P7_TransScalp"))
    results.append(backtest_p7(signals, z_entry=1.0, z_exit=0.5,
                               tx_cost_bps=tx_cost_bps,
                               name="P7_Loose"))
    results.append(backtest_p7(signals, z_entry=2.0, z_exit=0.2,
                               tx_cost_bps=tx_cost_bps,
                               name="P7_Tight"))

    # ── P8: IPES Adaptive Entry ─────────────────────────────────────
    results.append(backtest_p8(signals, tx_cost_bps=tx_cost_bps,
                               min_spread_exit=min_spread_exit,
                               name="P8_IPESAdaptive"))

    # ── P9: Cascade Fade ────────────────────────────────────────────
    results.append(backtest_p9(signals, dp_perp_min=1.3, E_perp_min=0.3,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P9_CascadeFade"))
    results.append(backtest_p9(signals, dp_perp_min=1.1, E_perp_min=0.2,
                               tx_cost_bps=tx_cost_bps, min_spread_exit=min_spread_exit,
                               name="P9_Aggressive"))

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
    Run bootstrap CI tests comparing strategies on drawdown and PnL.

    Compares P3 vs P2A, P5 vs P2A, P9 vs P2A on drawdown.
    PnL CIs for P2A, P3, P5, P7, P9.

    Returns dict with keys: 'drawdown_tests', 'pnl_tests'.
    """
    def make_p2a(sigs):
        return backtest_p2(sigs, has_stop_loss=False, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit)

    def make_p3(sigs):
        return backtest_p3(sigs, E_threshold=E_threshold, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit, name="P3_Full")

    def make_p5(sigs):
        return backtest_p5(sigs, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit, name="P5_OvershootFade")

    def make_p7(sigs):
        return backtest_p7(sigs, tx_cost_bps=tx_cost_bps, name="P7_TransScalp")

    def make_p9(sigs):
        return backtest_p9(sigs, tx_cost_bps=tx_cost_bps,
                          min_spread_exit=min_spread_exit, name="P9_CascadeFade")

    drawdown_tests = []
    # P3 vs P2A
    drawdown_tests.append(bootstrap_drawdown_ci(signals, make_p2a, make_p3,
                                                n_bootstrap=n_bootstrap))
    # P5 vs P2A
    drawdown_tests.append(bootstrap_drawdown_ci(signals, make_p2a, make_p5,
                                                n_bootstrap=n_bootstrap))
    # P9 vs P2A
    drawdown_tests.append(bootstrap_drawdown_ci(signals, make_p2a, make_p9,
                                                n_bootstrap=n_bootstrap))

    # PnL CIs
    pnl_tests = []
    for fn in [make_p2a, make_p3, make_p5, make_p7, make_p9]:
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
