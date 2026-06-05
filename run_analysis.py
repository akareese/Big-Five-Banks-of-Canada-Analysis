from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import BANKS, load_data

TRADING_DAYS = 252

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


COLORS = {
    "RY.TO": "#1F4E9C",
    "TD.TO": "#205E3B",
    "BNS.TO": "#E03131",
    "BMO.TO": "#399BEB",
    "CM.TO": "#8B0000",
}


def annualise_return(prices: pd.Series) -> float:
    years = len(prices) / TRADING_DAYS
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def main() -> None:
    prices, div_yields, source = load_data()
    rets = prices.pct_change().dropna()
    labels = {t: BANKS[t] for t in prices.columns}

    summary = pd.DataFrame({
        "Total Return": prices.iloc[-1] / prices.iloc[0] - 1,
        "Ann. Return": {t: annualise_return(prices[t]) for t in prices.columns},
        "Ann. Volatility": rets.std() * np.sqrt(TRADING_DAYS),
        "Dividend Yield": pd.Series(div_yields),
    })
    summary.index = [labels[t] for t in summary.index]

    fmt = (summary * 100).round(1).astype(str) + "%"
    print(f"\nData source: {source}")
    print(f"Period: {prices.index[0].date()} -> {prices.index[-1].date()}\n")
    print("Big Five Canadian Banks — comparison")
    print("=" * 64)
    print(fmt.to_string())
    print("=" * 64)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Big Five Canadian Banks  ({source} data, "
                 f"{prices.index[0].year}–{prices.index[-1].year})",
                 fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    growth = prices / prices.iloc[0]
    for t in prices.columns:
        ax.plot(growth[t], label=labels[t], color=COLORS[t], lw=1.8)
    ax.set_title("Growth of One Dollar Invested")
    ax.set_ylabel("Value (start = 1.0)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    for t in prices.columns:
        ax.scatter(summary.loc[labels[t], "Ann. Volatility"] * 100,
                   summary.loc[labels[t], "Ann. Return"] * 100,
                   s=140, color=COLORS[t], zorder=3)
        ax.annotate(labels[t].split("(")[0].strip(),
                    (summary.loc[labels[t], "Ann. Volatility"] * 100,
                     summary.loc[labels[t], "Ann. Return"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Risk vs Return (Annualised)")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Return (%)")

    ax = axes[1, 0]
    ynames = [labels[t] for t in prices.columns]
    yvals = [div_yields[t] * 100 for t in prices.columns]
    bars = ax.bar(ynames, yvals, color=[COLORS[t] for t in prices.columns])
    ax.set_title("Trailing Dividend Yield")
    ax.set_ylabel("Yield (%)")
    ax.tick_params(axis="x", rotation=25)
    for bar, v in zip(bars, yvals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f"{v:.1f}%", ha="center", fontsize=8)

    ax = axes[1, 1]
    corr = rets.corr()
    corr.index = [labels[t].split("(")[0].strip() for t in corr.index]
    corr.columns = corr.index
    im = ax.imshow(corr, cmap="YlGnBu", vmin=corr.values.min(), vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Daily-Return Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "banks_showdown.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved chart pack -> {out}")


if __name__ == "__main__":
    main()
