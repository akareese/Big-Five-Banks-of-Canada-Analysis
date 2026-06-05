from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BANKS, load_data

TRADING_DAYS = 252

COLORS = {
    "RY.TO": "#1F4E9C",
    "TD.TO": "#205E3B",
    "BNS.TO": "#E03131",
    "BMO.TO": "#399BEB",
    "CM.TO": "#8B0000",
}

st.set_page_config(
    page_title="Big Five Canadian Banks",
    page_icon="🍁",
    layout="wide",
)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching bank data…")
def get_data(start: str, end: str):
    return load_data(start, end)


def annualise_return(series: pd.Series) -> float:
    years = len(series) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


st.sidebar.header("Controls")

today = dt.date.today()

start_date = st.sidebar.date_input(
    "Start date", value=dt.date(2015, 1, 1),
    min_value=dt.date(2005, 1, 1), max_value=today,
)
end_date = st.sidebar.date_input(
    "End date", value=min(dt.date(2024, 12, 31), today),
    min_value=dt.date(2005, 1, 2), max_value=today,
)

selected = st.sidebar.multiselect(
    "Banks to compare",
    options=list(BANKS),
    default=list(BANKS),
    format_func=lambda t: BANKS[t],
)

st.title("🍁 Big Five Canadian Banks — Showdown")
st.write(
    "Compare **RBC, TD, Scotiabank, BMO, and CIBC** on long-run return, risk, "
    "dividend income, and how tightly they move together."
)

if not selected:
    st.warning("Pick at least one bank in the sidebar.")
    st.stop()
if start_date >= end_date:
    st.warning("Start date must be before end date.")
    st.stop()

prices_all, yields, source = get_data(str(start_date), str(end_date))
prices = prices_all[selected]
rets = prices.pct_change().dropna()
labels = {t: BANKS[t] for t in selected}

if source == "yfinance":
    st.success("Live data from Yahoo Finance", icon="✅")
else:
    st.info(
        "Showing **simulated** data — Yahoo Finance was unreachable or "
        "rate-limited. Numbers are illustrative, not real. Run locally with "
        "internet for live figures.",
        icon="ℹ️",
    )

summary = pd.DataFrame({
    "Total Return": prices.iloc[-1] / prices.iloc[0] - 1,
    "Ann. Return": {t: annualise_return(prices[t]) for t in selected},
    "Ann. Volatility": rets.std() * np.sqrt(TRADING_DAYS),
    "Dividend Yield": pd.Series({t: yields[t] for t in selected}),
})

best = summary["Total Return"].idxmax()
c1, c2, c3 = st.columns(3)
c1.metric("Best total return", labels[best],
          f"{summary.loc[best, 'Total Return'] * 100:.0f}%")
c2.metric("Highest dividend yield",
          labels[summary["Dividend Yield"].idxmax()],
          f"{summary['Dividend Yield'].max() * 100:.1f}%")
c3.metric("Avg. pairwise correlation",
          f"{rets.corr().where(~np.eye(len(selected), dtype=bool)).stack().mean():.2f}"
          if len(selected) > 1 else "—")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Growth of One Dollar Invested")
    growth = prices / prices.iloc[0]
    fig = go.Figure()
    for t in selected:
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[t], name=labels[t],
            line=dict(color=COLORS[t], width=2),
        ))
    fig.update_layout(
        yaxis_title="Value (start = $1)", hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), legend_title_text="",
        height=380,
    )
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Risk vs Return (Annualised)")
    fig = go.Figure()
    for t in selected:
        fig.add_trace(go.Scatter(
            x=[summary.loc[t, "Ann. Volatility"] * 100],
            y=[summary.loc[t, "Ann. Return"] * 100],
            mode="markers+text", name=labels[t],
            text=[labels[t]], textposition="top center",
            marker=dict(size=16, color=COLORS[t]),
            showlegend=False,
        ))
    fig.update_layout(
        xaxis_title="Volatility (%)", yaxis_title="Return (%)",
        margin=dict(l=10, r=10, t=10, b=10), height=380,
    )
    st.plotly_chart(fig, width='stretch')

left2, right2 = st.columns(2)

with left2:
    st.subheader("Trailing Dividend Yield")
    yvals = [yields[t] * 100 for t in selected]
    fig = go.Figure(go.Bar(
        x=[labels[t] for t in selected], y=yvals,
        marker_color=[COLORS[t] for t in selected],
        text=[f"{v:.1f}%" for v in yvals], textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Yield (%)", margin=dict(l=10, r=10, t=10, b=10),
        height=360,
    )
    st.plotly_chart(fig, width='stretch')

with right2:
    st.subheader("Daily-Return Correlation")
    if len(selected) > 1:
        corr = rets.corr()
        short = [labels[t].split("(")[0].strip() for t in corr.columns]
        fig = px.imshow(
            corr.values, x=short, y=short,
            color_continuous_scale="YlGnBu", zmin=corr.values.min(), zmax=1,
            text_auto=".2f", aspect="auto",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption("Select two or more banks to see correlations.")

st.divider()
st.subheader("Comparison table")
display = summary.copy()
display.index = [labels[t] for t in display.index]
st.dataframe(
    display.style.format({
        "Total Return": "{:.1%}", "Ann. Return": "{:.1%}",
        "Ann. Volatility": "{:.1%}", "Dividend Yield": "{:.1%}",
    }),
    width='stretch',
)

with st.expander("💡 The interesting takeaway"):
    st.write(
        "The correlation panel is the real story: the Big Five move together so "
        "tightly (typically 0.7–0.9 daily-return correlation) that holding all "
        "five gives far less diversification than it looks like — you mostly own "
        "one big bet on Canadian banking."
    )

st.caption(
    f"Data source: {source} · {prices.index[0].date()} → {prices.index[-1].date()}"
)
