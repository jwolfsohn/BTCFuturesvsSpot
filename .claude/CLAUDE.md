# Claude Custom Instructions for BTC-Perp-Spot-IPES

## Core Mandate: Do No Harm to the Research

- This project applies the IPES framework (Shen, Huang, Zuo, Zivot 2026) to BTC perpetual futures vs BTC spot on Binance.
- The core VECM → Ψ(1) → ψ → d^P → E_i → IPES math is identical to the stablecoin reference project. Do NOT alter it.
- Do NOT remove the cointegration prerequisite for IPES. IPES mathematically requires a VECM, which requires cointegration.
- Do NOT suggest standard stop-losses as improvements over the E_i filter. The literature shows price-based stop-losses hurt crypto pairs trading.
- Binance futures fees are 1.5–4.5 bps taker, 0–1.8 bps maker. Do NOT use equities fee assumptions.
- Only make changes when 100% certain they are mathematically sound. If ambiguous, ASK FIRST.

## Key Differences from the Stablecoin Reference Project

1. **Series construction**: `log(BTC_spot)` and `log(BTC_perp)` directly — NO cross-rate construction needed (both are USDT-denominated).
2. **"Depeg leg" → "dislocation leg"**: Which market (spot=0 or perp=1) is driving the basis apart. Measured by deviation from the pair's log-price midpoint, not from $1.
3. **Stop-losses**: Percentage-based (e.g., -2%, -5%), not dollar-level ($0.90). BTC trades at ~$40K–100K.
4. **Max holding period**: 24 hours to prevent funding rate bleed on perp positions.
5. **Funding rate data**: Downloaded for context and position cost tracking. NOT fed into the VECM.

## Terminology Mapping

| Stablecoin Project | This Project |
|---|---|
| depeg | basis dislocation |
| depeg leg | dislocation leg (spot or perp) |
| stablecoin near $1 | BTC at market price |
| dollar-level stop ($0.90) | percentage stop (e.g., -2%) |
| SC/USD price | BTC spot close |
| USDT/USD price | BTC perp close |

## Verification Checklist (Before Any Commit)

> "Am I sure this change preserves the VECM → Ψ(1) → d^P → E_i → IPES math exactly?"
> "Does β = (1,-1)' make economic sense for spot vs perp?" (Yes — funding rate mechanism.)
> "Are transaction costs realistic for Binance futures?"
> "Is the walk-forward estimation free of look-ahead bias?"
