# Useful Commands for BTC IPES Pipeline

This document provides a quick reference for running the various scripts in the BTC Futures vs Spot pipeline. All commands should be run from the root of the repository (`/Users/jackwolfsohn/BTCFuturesvsSpot`).

## Pipeline Execution

### 1. Download Data
Downloads historical data from Binance for BTC Spot and Perpetual Futures.

```bash
# Standard download
python3 -m scripts.download_data

# Download 1s data for paper replication (Jan 1-5, 2024)
python3 -m scripts.download_data --paper

# Download extended data: 2020-2024 for multi-regime testing
python3 -m scripts.download_data --all
```

### 2. Validate Data
Loads the downloaded data, prints validation summaries (rows, gaps, zero-volume stats), and aligns spot vs. perpetual futures along with the funding rate.

```bash
python3 -m scripts.load_data
```

### 3. Quick Test on Small Slice
Runs the VECM + IPES estimation on a small predefined slice of data to act as a quick sanity check without running the full dataset.

```bash
python3 -m scripts.vecm_ipes
```

### 4. Full Pipeline
Runs the complete VECM estimation and portfolio backtests.

```bash
# Full pipeline (~1-2 hours)
python3 -m scripts.run_pipeline

# Fast run (Skips Part 2: 1-second cascade deep-dives)
python3 -m scripts.run_pipeline --fast

# Skip VECM entirely and use stub signals (Implies --fast)
python3 -m scripts.run_pipeline --no-vecm
```

### 5. Plot Results
Generates plots for paper results based on the output of the full pipeline.

```bash
python3 -m scripts.plot_paper_results
```
