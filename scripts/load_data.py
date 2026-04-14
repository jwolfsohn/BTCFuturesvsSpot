"""
In-memory data loader for Binance kline and funding rate zips.

Handles BTC spot klines, BTC perpetual futures klines, and funding rate data.
No extraction to disk — reads CSVs directly from zip archives.

Usage
-----
    from scripts.load_data import load_spot, load_perp, load_aligned_pair

    spot = load_spot()                          # raw DataFrame
    spot_c, perp_c = load_aligned_pair()        # aligned close Series
    spot_c, perp_c, fr = load_with_funding()    # + funding rate
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent.parent / "data" / "binance"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

PRICE_COLS = ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]

FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


# ── Low-level readers ──────────────────────────────────────────────────────


def _read_zip(path: Path) -> pd.DataFrame:
    """Read the single CSV inside a Binance kline zip into a DataFrame.

    Handles both headerless (spot) and headered (futures) CSV formats.
    Futures CSVs use slightly different column names (count vs trades,
    taker_buy_volume vs taker_buy_base) — we normalize to KLINE_COLS.
    """
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw_bytes = z.read(name)

    # Detect header: if first line starts with 'open_time', it has a header
    first_line = raw_bytes[:200].decode("utf-8", errors="ignore").split("\n")[0]
    if first_line.startswith("open_time"):
        df = pd.read_csv(io.BytesIO(raw_bytes), header=0)
        # Normalize column names (futures uses 'count', 'taker_buy_volume', etc.)
        rename_map = {
            "count": "trades",
            "taker_buy_volume": "taker_buy_base",
            "taker_buy_quote_volume": "taker_buy_quote",
        }
        df = df.rename(columns=rename_map)
        # Drop any extra columns not in KLINE_COLS
        df = df[[c for c in KLINE_COLS if c in df.columns]]
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=KLINE_COLS)

    df[PRICE_COLS] = df[PRICE_COLS].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").drop(columns=["close_time", "ignore"])
    return df


def _read_funding_zip(path: Path) -> pd.DataFrame:
    """Read a Binance funding rate zip into a DataFrame."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw_bytes = z.read(name)

    # Detect header
    first_line = raw_bytes[:100].decode("utf-8", errors="ignore").split("\n")[0]
    if first_line.startswith("calc_time"):
        df = pd.read_csv(io.BytesIO(raw_bytes), header=0)
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=FUNDING_COLS)

    df["last_funding_rate"] = df["last_funding_rate"].astype(float)
    df["funding_interval_hours"] = df["funding_interval_hours"].astype(float)
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("calc_time")
    return df


def _read_aggtrades_zip(path: Path) -> pd.DataFrame:
    """Read a Binance aggTrades zip into a DataFrame."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw_bytes = z.read(name)

    first_line = raw_bytes[:200].decode("utf-8", errors="ignore").split("\n")[0]
    if first_line.startswith("agg_trade_id"):
        df = pd.read_csv(io.BytesIO(raw_bytes), header=0)
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None,
                         names=["agg_trade_id", "price", "quantity",
                                "first_trade_id", "last_trade_id",
                                "transact_time", "is_buyer_maker"])

    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.set_index("transact_time")
    return df


def _resample_aggtrades_to_1s(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Resample aggTrades to 1-second OHLCV bars matching the kline format.

    Returns a DataFrame with the same columns as _read_zip() output
    (indexed by open_time, with standard kline columns).
    """
    ohlcv = trades["price"].resample("1s").ohlc()
    ohlcv.columns = ["open", "high", "low", "close"]
    ohlcv["volume"] = trades["quantity"].resample("1s").sum()
    ohlcv["quote_volume"] = (trades["price"] * trades["quantity"]).resample("1s").sum()
    ohlcv["trades"] = trades["price"].resample("1s").count()

    # Taker buy = where is_buyer_maker is False (taker is buyer)
    taker_buy_mask = ~trades["is_buyer_maker"]
    ohlcv["taker_buy_base"] = trades.loc[taker_buy_mask, "quantity"].resample("1s").sum()
    ohlcv["taker_buy_quote"] = (
        trades.loc[taker_buy_mask, "price"] * trades.loc[taker_buy_mask, "quantity"]
    ).resample("1s").sum()

    # Drop seconds with no trades (NaN from ohlc)
    ohlcv = ohlcv.dropna(subset=["open"])
    ohlcv = ohlcv.fillna(0.0)  # fill NaN taker_buy cols for seconds with no taker buys

    ohlcv["trades"] = ohlcv["trades"].astype(int)
    ohlcv.index.name = "open_time"
    return ohlcv


def load_futures_1s_from_aggtrades(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Load futures 1s bars by resampling aggTrades.

    Used because Binance Data Vision doesn't provide 1s klines for futures.
    AggTrades are stored in data/binance/futures/{symbol}/aggTrades/.
    """
    folder = DATA_ROOT / "futures" / symbol / "aggTrades"
    zips = sorted(folder.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No aggTrades zips found in {folder}")

    frames = []
    for p in zips:
        trades = _read_aggtrades_zip(p)
        bars = _resample_aggtrades_to_1s(trades)
        frames.append(bars)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ── Kline loaders ──────────────────────────────────────────────────────────


def load_klines(market: str, symbol: str, interval: str) -> pd.DataFrame:
    """
    Load and concatenate all kline zips for a market/symbol/interval.

    Parameters
    ----------
    market   : 'spot' or 'futures'
    symbol   : e.g. 'BTCUSDT'
    interval : e.g. '1m'

    Returns
    -------
    DataFrame indexed by UTC open_time, sorted ascending, duplicates dropped.
    """
    folder = DATA_ROOT / market / symbol / interval
    zips = sorted(folder.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip files found in {folder}")

    frames = [_read_zip(p) for p in zips]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_spot(symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
    """Convenience: load spot klines."""
    return load_klines("spot", symbol, interval)


def load_perp(symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
    """Convenience: load perpetual futures klines.

    For 1s interval, falls back to aggTrades resampling if klines aren't available.
    """
    if interval == "1s":
        kline_folder = DATA_ROOT / "futures" / symbol / "1s"
        if not list(kline_folder.glob("*.zip")) if kline_folder.exists() else True:
            return load_futures_1s_from_aggtrades(symbol)
    return load_klines("futures", symbol, interval)


def load_close(market: str, symbol: str, interval: str) -> pd.Series:
    """Return just the close price series for a market/symbol/interval."""
    return load_klines(market, symbol, interval)["close"].rename(f"{symbol}_{market}")


# ── Funding rate loader ────────────────────────────────────────────────────


def load_funding_rate(symbol: str = "BTCUSDT") -> pd.Series:
    """
    Load all funding rate zips, return Series indexed by UTC calc_time.

    Returns
    -------
    pd.Series of funding rates (decimal, e.g. 0.0001 = 1 bp).
    """
    folder = DATA_ROOT / "futures" / symbol / "fundingRate"
    zips = sorted(folder.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No funding rate zips found in {folder}")

    frames = [_read_funding_zip(p) for p in zips]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df["last_funding_rate"].rename("funding_rate")


# ── Aligned pair loaders ───────────────────────────────────────────────────


def load_aligned_pair(
    symbol: str = "BTCUSDT", interval: str = "1m",
) -> tuple[pd.Series, pd.Series]:
    """
    Load spot and perp close prices, aligned to common timestamps.

    Returns (spot_close, perp_close) as named pd.Series ('BTC_SPOT', 'BTC_PERP').
    Inner join — only timestamps present in both are kept.
    """
    spot_df = load_spot(symbol, interval)
    perp_df = load_perp(symbol, interval)

    spot_close = spot_df["close"].rename("BTC_SPOT")
    perp_close = perp_df["close"].rename("BTC_PERP")

    both = pd.concat([spot_close, perp_close], axis=1, join="inner").dropna()
    return both["BTC_SPOT"], both["BTC_PERP"]


def load_with_funding(
    symbol: str = "BTCUSDT", interval: str = "1m",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Like load_aligned_pair but also returns funding rate
    forward-filled to the kline frequency.

    Returns (spot_close, perp_close, funding_rate).
    """
    spot_close, perp_close = load_aligned_pair(symbol, interval)
    funding = load_funding_rate(symbol)

    # Reindex funding to the kline timestamps and forward-fill
    funding_1m = funding.reindex(spot_close.index, method="ffill")
    funding_1m = funding_1m.fillna(0.0)  # fill any leading NaNs

    return spot_close, perp_close, funding_1m


# ── Data validation ────────────────────────────────────────────────────────


def _expected_freq(interval: str) -> pd.Timedelta:
    """Map interval string to expected timedelta between rows."""
    return {"1s": pd.Timedelta(seconds=1), "1m": pd.Timedelta(minutes=1)}[interval]


def validate(df: pd.DataFrame, name: str, interval: str,
             gap_threshold: int = 3, quiet: bool = False) -> dict:
    """
    Run data quality checks. Returns a report dict. Prints summary unless quiet.

    Parameters
    ----------
    df             : DataFrame from load_klines()
    name           : descriptive name, e.g. 'BTC_SPOT'
    interval       : '1m' or '1s'
    gap_threshold  : flag gaps longer than this many expected periods
    """
    freq = _expected_freq(interval)
    diffs = df.index.to_series().diff()
    gap_limit = freq * gap_threshold
    gap_mask = diffs > gap_limit
    gaps = [(idx, diffs[idx]) for idx in diffs[gap_mask].index]

    report = {
        "name": name,
        "interval": interval,
        "rows": len(df),
        "range_start": df.index[0],
        "range_end": df.index[-1],
        "zero_volume": int((df["volume"] == 0).sum()),
        "nan_close": int(df["close"].isna().sum()),
        "bad_close": int(((df["close"] <= 0) | np.isinf(df["close"])).sum()),
        "gaps": gaps,
    }

    if not quiet:
        _print_report(report)

    return report


def _print_report(r: dict) -> None:
    """Pretty-print a validation report."""
    print(f"\n{'=' * 50}")
    print(f"  {r['name']} {r['interval']}")
    print(f"{'=' * 50}")
    print(f"  Rows:        {r['rows']:,}")
    print(f"  Range:       {r['range_start']} -> {r['range_end']}")
    print(f"  Zero-volume: {r['zero_volume']:,}"
          f" ({r['zero_volume'] / r['rows'] * 100:.1f}%)")
    if r["nan_close"] or r["bad_close"]:
        print(f"  NaN close:   {r['nan_close']}")
        print(f"  Bad close:   {r['bad_close']}")
    else:
        print(f"  Close:       all valid (no NaN/Inf/<=0)")
    if r["gaps"]:
        print(f"  Gaps (>{r['interval']} x {3}):")
        for ts, dur in r["gaps"][:10]:
            print(f"    {ts}  gap = {dur}")
        if len(r["gaps"]) > 10:
            print(f"    ... and {len(r['gaps']) - 10} more")
    else:
        print(f"  Gaps:        none")


def extract_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a DataFrame to a UTC date range (inclusive)."""
    return df.loc[start:end]


# ── Main: validate all data ────────────────────────────────────────────────


if __name__ == "__main__":
    print("Loading BTC spot klines...")
    spot = load_spot()
    spot_report = validate(spot, "BTC_SPOT", "1m")

    print("\nLoading BTC perpetual futures klines...")
    perp = load_perp()
    perp_report = validate(perp, "BTC_PERP", "1m")

    print("\nLoading funding rate...")
    try:
        fr = load_funding_rate()
        print(f"  Funding rate: {len(fr):,} observations")
        print(f"  Range: {fr.index[0]} -> {fr.index[-1]}")
        print(f"  Mean: {fr.mean():.6f}  Std: {fr.std():.6f}")
        print(f"  Min:  {fr.min():.6f}  Max: {fr.max():.6f}")
    except FileNotFoundError as e:
        print(f"  Funding rate not available: {e}")

    print("\nLoading aligned pair...")
    spot_c, perp_c = load_aligned_pair()
    print(f"  Aligned timestamps: {len(spot_c):,}")
    print(f"  Range: {spot_c.index[0]} -> {spot_c.index[-1]}")

    basis = spot_c - perp_c
    print(f"\n  Basis (spot - perp) summary:")
    print(f"    Mean:  {basis.mean():+.4f}")
    print(f"    Std:   {basis.std():.4f}")
    print(f"    Min:   {basis.min():+.4f}")
    print(f"    Max:   {basis.max():+.4f}")
