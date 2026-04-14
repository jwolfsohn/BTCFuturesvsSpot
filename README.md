# BTC Perpetual Futures vs Spot -- IPES Trading Strategy

Applies the IPES (Instantaneous Pricing Efficiency Share) framework from
Shen, Huang, Zuo, Zivot (2026) to BTC spot vs BTC perpetual futures on Binance.

## Mathematical Pipeline

Rolling VECM -> Psi(1) -> Price Discovery Beta d^P -> Pricing Error E_i -> IPES Score
-> Permanent/Transitory Shock Decomposition -> Trading Signals

## Setup

Python 3.10+

```bash
pip install numpy pandas statsmodels scipy requests
```

## Usage

```bash
# 1. Download data (~30 min)
python3 -m scripts.download_data

# 2. Validate data
python3 -m scripts.load_data

# 3. Quick test on small slice
python3 -m scripts.vecm_ipes

# 4. Full pipeline (~1-2 hours with robustness)
python3 -m scripts.run_pipeline

# 5. Quick run without robustness sweep (~15 min)
python3 -m scripts.run_pipeline --skip-robustness
```

## Default Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| W | 360 | 6-hour window at 1min |
| K | 5 | VAR(5) -> VECM(4) |
| deterministic | "ci" | Case 2 (constant in CI eq) |
| constrain_beta | True | beta=(1,-1)' enforced by funding |
| E_threshold | 0.5 | Swept: {0.3, 0.5, 0.75, 1.0, 1.5} |
| tx_cost_bps | 5 | Swept: {0, 2, 5, 10, 15} |
| z_entry | 2.0 | LowZ variants test {0.5, 1.0} |
| max_hold_signals | 96 | 24 hours (96 x 15min) |

## Reference

- IPES paper: Shen, Huang, Zuo, Zivot (2026) -- "Identifying Price Discovery Without Trade Direction"
- Stablecoin research project: ~/CryptoTradingResearch
