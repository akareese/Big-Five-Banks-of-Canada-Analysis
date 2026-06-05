from __future__ import annotations

import numpy as np
import pandas as pd


BANKS = {
    "RY.TO": "Royal Bank (RBC)",
    "TD.TO": "TD Bank",
    "BNS.TO": "Scotiabank",
    "BMO.TO": "BMO",
    "CM.TO": "CIBC",
}

_PROFILE = {
    "RY.TO": (0.09, 0.20, 0.038),
    "TD.TO": (0.08, 0.21, 0.045),
    "BNS.TO": (0.06, 0.22, 0.058),
    "BMO.TO": (0.08, 0.21, 0.046),
    "CM.TO": (0.07, 0.23, 0.052),
}


def _simulate(start: str, end: str, seed: int = 6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    tickers = list(BANKS)

    mu = np.array([_PROFILE[t][0] for t in tickers])
    sigma = np.array([_PROFILE[t][1] for t in tickers])
    yields = {t: _PROFILE[t][2] for t in tickers}

    dt = 1 / 252

    sector = rng.standard_normal((n, 1))
    idio = rng.standard_normal((n, len(tickers)))
    shocks = 0.55 * sector + 0.30 * idio

    daily = mu * dt + sigma * np.sqrt(dt) * shocks
    prices = 50 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers), yields, "simulated"


def load_data(start: str = "2015-01-01", end: str = "2024-12-31"):
    try:
        import yfinance as yf

        tickers = list(BANKS)
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"] if "Close" in raw else raw
        prices = prices.dropna(how="all").ffill().dropna()

        if not prices.empty:
            yields = {}
            for t in tickers:
                divs = yf.Ticker(t).dividends
                last_year = divs[divs.index >= divs.index.max() - pd.Timedelta(days=365)]
                yields[t] = float(last_year.sum() / prices[t].iloc[-1])
            return prices[tickers], yields, "yfinance"
    except Exception:
        pass

    return _simulate(start, end)
