"""
Download Binance historical data for BTC spot klines, BTC perpetual futures
klines, and funding rate from Binance Data Vision.
No API key or account required — uses the free public data repository.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from pathlib import Path


BASE_URL_SPOT = "https://data.binance.vision/data/spot"
BASE_URL_FUTURES = "https://data.binance.vision/data/futures/um"


def _session_with_retries(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Create a requests session with automatic retry on connection errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_session = _session_with_retries()


def _download_file(url: str, output_path: Path) -> bool:
    """Download a single file with retry logic. Returns True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True

    try:
        response = _session.get(url, stream=True, timeout=30)
    except requests.exceptions.ConnectionError as e:
        print(f"  Connection error: {output_path.name} ({e})")
        return False

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return True
    else:
        print(f"  Not available: {output_path.name} (HTTP {response.status_code})")
        return False


# ── Spot klines ─────────────────────────────────────────────────────────────


def download_monthly_klines_spot(symbol: str, interval: str, year: int, month: int, output_dir: str):
    """Download a single monthly spot kline file."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL_SPOT}/monthly/klines/{symbol}/{interval}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


def download_daily_klines_spot(symbol: str, interval: str, date_str: str, output_dir: str):
    """Download a single daily spot kline file. date_str format: YYYY-MM-DD."""
    filename = f"{symbol}-{interval}-{date_str}.zip"
    url = f"{BASE_URL_SPOT}/daily/klines/{symbol}/{interval}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


# ── Futures klines ──────────────────────────────────────────────────────────


def download_monthly_klines_futures(symbol: str, interval: str, year: int, month: int, output_dir: str):
    """Download a single monthly USD-M perpetual futures kline file."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL_FUTURES}/monthly/klines/{symbol}/{interval}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


def download_daily_klines_futures(symbol: str, interval: str, date_str: str, output_dir: str):
    """Download a single daily USD-M perpetual futures kline file. date_str: YYYY-MM-DD."""
    filename = f"{symbol}-{interval}-{date_str}.zip"
    url = f"{BASE_URL_FUTURES}/daily/klines/{symbol}/{interval}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


# ── Futures aggTrades (for constructing 1s bars) ──────────────────────────


def download_daily_aggtrades_futures(symbol: str, date_str: str, output_dir: str):
    """Download a single daily USD-M futures aggTrades file. date_str: YYYY-MM-DD."""
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL_FUTURES}/daily/aggTrades/{symbol}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


# ── Funding rate ────────────────────────────────────────────────────────────


def download_monthly_funding_rate(symbol: str, year: int, month: int, output_dir: str):
    """Download a single monthly funding rate file."""
    filename = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
    url = f"{BASE_URL_FUTURES}/monthly/fundingRate/{symbol}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


def download_daily_funding_rate(symbol: str, date_str: str, output_dir: str):
    """Download a single daily funding rate file. date_str: YYYY-MM-DD."""
    filename = f"{symbol}-fundingRate-{date_str}.zip"
    url = f"{BASE_URL_FUTURES}/daily/fundingRate/{symbol}/{filename}"
    return _download_file(url, Path(output_dir) / filename)


# ── Generic daily range downloader ──────────────────────────────────────────


def download_date_range_daily(download_fn, symbol, interval_or_none, start: str, end: str, output_dir: str):
    """
    Download daily files for a date range using the given download function.

    download_fn signature must be one of:
      - download_daily_klines_spot(symbol, interval, date_str, output_dir)
      - download_daily_klines_futures(symbol, interval, date_str, output_dir)
      - download_daily_funding_rate(symbol, date_str, output_dir)

    If interval_or_none is None, calls download_fn(symbol, date_str, output_dir).
    Otherwise calls download_fn(symbol, interval_or_none, date_str, output_dir).
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    current = start_dt

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        if interval_or_none is None:
            download_fn(symbol, date_str, output_dir)
        else:
            download_fn(symbol, interval_or_none, date_str, output_dir)
        current += timedelta(days=1)


def download_1s_klines_date_range(
    symbol: str, start: str, end: str,
) -> None:
    """
    Download 1-second data for both spot and futures over a date range.

    Spot: direct 1s klines from Binance Data Vision.
    Futures: aggTrades (1s klines not available), resampled to 1s in load_data.

    Storage: data/binance/spot/{symbol}/1s/
             data/binance/futures/{symbol}/aggTrades/

    Parameters
    ----------
    symbol : e.g. 'BTCUSDT'
    start  : 'YYYY-MM-DD'
    end    : 'YYYY-MM-DD'
    """
    spot_dir = f"data/binance/spot/{symbol}/1s"
    aggtrades_dir = f"data/binance/futures/{symbol}/aggTrades"

    print(f"\nDownloading {symbol} spot 1s klines ({start} to {end})...")
    download_date_range_daily(
        download_daily_klines_spot, symbol, "1s", start, end, spot_dir,
    )

    print(f"\nDownloading {symbol} futures aggTrades ({start} to {end})...")
    print("  (Binance doesn't provide futures 1s klines; using aggTrades → 1s resample)")
    download_date_range_daily(
        download_daily_aggtrades_futures, symbol, None, start, end, aggtrades_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--paper", action="store_true",
                        help="Download 1s data for paper replication (Jan 1-5, 2024)")
    parser.add_argument("--all", action="store_true",
                        help="Download extended data: 2020-2024 for multi-regime testing")
    args = parser.parse_args()

    if args.paper:
        # Paper replication: 1s data for Jan 1-5, 2024 (Section 5.2 cascade on Jan 3)
        download_1s_klines_date_range("BTCUSDT", "2024-01-01", "2024-01-05")

        # Also download funding rate for context
        print("\nDownloading BTC funding rate (Jan 2024)...")
        download_monthly_funding_rate("BTCUSDT", 2024, 1,
                                      "data/binance/futures/BTCUSDT/fundingRate")
    else:
        # --all: 2020-2024, default: 2024 only
        years = range(2020, 2025) if args.all else [2024]
        months_list = [(y, m) for y in years for m in range(1, 13)]

        label = "2020-2024" if args.all else "2024"
        print(f"\nDownloading BTC spot 1m monthly klines ({label})...")
        for year, month in months_list:
            download_monthly_klines_spot("BTCUSDT", "1m", year, month,
                                         "data/binance/spot/BTCUSDT/1m")

        print(f"\nDownloading BTC perpetual futures 1m monthly klines ({label})...")
        for year, month in months_list:
            download_monthly_klines_futures("BTCUSDT", "1m", year, month,
                                            "data/binance/futures/BTCUSDT/1m")

        print(f"\nDownloading BTC funding rate ({label})...")
        for year, month in months_list:
            download_monthly_funding_rate("BTCUSDT", year, month,
                                          "data/binance/futures/BTCUSDT/fundingRate")

    print("\nAll downloads complete!")
