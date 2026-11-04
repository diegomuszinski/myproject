"""Streamlit dashboard for the FINOR index-tracking analysis."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import nnls


SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT",
    "AMZN", "ANET", "AVGO", "AXP", "BA", "BAC", "BKNG", "BLK", "BMY",
    "BNY", "BRK-B", "C", "CAT", "CMCSA", "COF", "COP", "COST", "CRM",
    "CSCO", "CVS", "CVX", "DE", "DELL", "DHR", "DIS", "DUK", "EMR",
    "FDX", "GD", "GE", "GEV", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD",
    "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY",
    "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "META", "MMM",
    "MO", "MRK", "MS", "MSFT", "MU", "NEE", "NFLX", "NOW", "NVDA", "ORCL",
    "PANW", "PEP", "PFE", "PG", "PLTR", "PM", "QCOM", "RTX", "SBUX", "SCHW",
    "SNDK", "SO", "T", "TMO", "TMUS", "TSLA", "TXN", "UBER", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

IBOV = [
    "ALOS3.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AXIA3.SA", "AZZA3.SA",
    "B3SA3.SA", "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA",
    "BRAV3.SA", "BPAC11.SA", "CXSE3.SA", "CEAB3.SA", "CMIG4.SA", "COGN3.SA",
    "CSMG3.SA", "CPLE3.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA", "CURY3.SA",
    "CYRE3.SA", "DIRR3.SA", "EMBJ3.SA", "ENGI11.SA", "ENEV3.SA", "EGIE3.SA",
    "EQTL3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "HAPV3.SA", "HYPE3.SA",
    "IGTI11.SA", "ISAE4.SA", "ITSA4.SA", "ITUB4.SA", "KLBN11.SA", "RENT3.SA",
    "LREN3.SA", "MGLU3.SA", "POMO4.SA", "MBRF3.SA", "BEEF3.SA", "MOTV3.SA",
    "MRVE3.SA", "MULT3.SA", "NATU3.SA", "PETR3.SA", "PETR4.SA", "PSSA3.SA",
    "PRIO3.SA", "RADL3.SA", "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "SANB11.SA",
    "CSNA3.SA", "SMFT3.SA", "SUZB3.SA", "TAEE11.SA", "VIVT3.SA", "TEND3.SA",
    "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VAMO3.SA",
    "VBBR3.SA", "VIVA3.SA", "WEGE3.SA", "YDUQ3.SA",
]

MARKETS = {
    "S&P 100": {"benchmark": "^OEX", "tickers": SP100, "currency": "USD"},
    "Ibovespa": {"benchmark": "^BVSP", "tickers": IBOV, "currency": "BRL"},
}


st.set_page_config(page_title="FINOR | Index Tracking", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], benchmark: str, start: date, end: date):
    symbols = list(dict.fromkeys([*tickers, benchmark]))
    raw = yf.download(
        symbols,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("O Yahoo Finance não retornou dados para o período escolhido.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns={"Close": symbols[0]}) if not isinstance(raw.columns, pd.MultiIndex) else close
    available = [t for t in tickers if t in close.columns and close[t].notna().sum() >= 30]
    if benchmark not in close.columns or close[benchmark].dropna().empty:
        raise ValueError(f"Não foi possível obter a série do benchmark {benchmark}.")
    if not available:
        raise ValueError("Nenhum componente do índice possui dados suficientes.")
    return close[available + [benchmark]].sort_index().ffill(limit=2), available


def tracking_portfolio(returns: pd.DataFrame, benchmark: str, k: int):
    clean = returns.dropna()
    if len(clean) < 30:
        raise ValueError("Há poucas observações comuns para estimar a carteira.")
    assets = [c for c in clean.columns if c != benchmark]
    coefficients, _ = nnls(clean[assets].to_numpy(), clean[benchmark].to_numpy())
    ranked = np.argsort(coefficients)[::-1][: min(k, len(assets))]
    selected = [assets[i] for i in ranked if coefficients[i] > 0]
    if not selected:
        selected = assets[: min(k, len(assets))]
    weights = coefficients[[assets.index(t) for t in selected]]
    if weights.sum() <= 0:
        weights = np.ones(len(selected))
    weights = weights / weights.sum()
    portfolio = clean[selected].mul(weights, axis=1).sum(axis=1).rename("Carteira")
    return pd.Series(weights, index=selected, name="Peso"), portfolio


def annualized_metrics(portfolio: pd.Series, benchmark: pd.Series):
    pair = pd.concat([portfolio, benchmark.rename("Índice")], axis=1).dropna()
    error = pair["Carteira"] - pair["Índice"]
    years = max(len(pair) / 252, 1 / 252)
    return {
        "Tracking error": error.std() * np.sqrt(252),
        "Correlação": pair["Carteira"].corr(pair["Índice"]),
        "Retorno carteira": (1 + pair["Carteira"]).prod() ** (1 / years) - 1,
        "Retorno índice": (1 + pair["Índice"]).prod() ** (1 / years) - 1,
        "Observações": len(pair),
    }


st.title("FINOR — Index Tracking")
st.caption("Dashboard interativo baseado no notebook de acompanhamento do S&P 100 e do Ibovespa.")

with st.sidebar:
    st.header("Configuração")
    market_name = st.selectbox("Índice", list(MARKETS))
    market = MARKETS[market_name]
    default_start = date(2019, 9, 30)
    start = st.date_input("Data inicial", default_start, min_value=date(2010, 1, 1))
    end = st.date_input("Data final", date.today(), min_value=start)
    k = st.slider("Número de ações (K)", 3, 30, 10)
    st.caption("As cotações são ajustadas pelo Yahoo Finance e podem mudar após revisões da fonte.")

if start >= end:
    st.error("A data final precisa ser posterior à data inicial.")
    st.stop()

try:
    with st.spinner("Baixando cotações e preparando as séries..."):
        prices, available = load_prices(tuple(market["tickers"]), market["benchmark"], start, end)
except Exception as exc:
    st.error(f"Falha ao carregar os dados: {exc}")
    st.stop()

returns = prices.pct_change(fill_method=None).dropna(how="all")
benchmark_returns = returns[market["benchmark"]].dropna()
usable = returns.dropna(subset=[market["benchmark"]])

try:
    weights, portfolio_returns = tracking_portfolio(usable, market["benchmark"], k)
    metrics = annualized_metrics(portfolio_returns, benchmark_returns)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

tab_overview, tab_portfolio, tab_data = st.tabs(["Visão geral", "Carteira", "Dados"])

with tab_overview:
    st.subheader(f"{market_name}: desempenho acumulado")
    levels = (1 + returns[[market["benchmark"]]].fillna(0)).cumprod()
    levels.columns = [market_name]
    levels["Carteira selecionada"] = (1 + portfolio_returns).cumprod()
    st.line_chart(levels)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracking error anualizado", f"{metrics['Tracking error']:.2%}")
    c2.metric("Correlação", f"{metrics['Correlação']:.3f}")
    c3.metric("Retorno anualizado", f"{metrics['Retorno carteira']:.2%}")
    c4.metric("Ações disponíveis", len(available))

with tab_portfolio:
    st.subheader(f"Carteira long-only com {len(weights)} ações")
    st.caption("Os pesos são estimados por mínimos quadrados não negativos sobre os retornos diários e normalizados para somar 100%.")
    weight_table = weights.sort_values(ascending=False).to_frame("Peso")
    weight_table["Peso"] = weight_table["Peso"].map(lambda x: f"{x:.2%}")
    st.dataframe(weight_table, use_container_width=True)
    st.bar_chart(weights.sort_values(ascending=True))
    residual = (portfolio_returns - benchmark_returns).rename("Erro diário")
    st.subheader("Erro de acompanhamento")
    st.line_chart(residual.to_frame())

with tab_data:
    st.subheader("Cobertura dos dados")
    coverage = prices.notna().sum().sort_values().to_frame("Pregões com dados")
    coverage["Cobertura"] = (coverage["Pregões com dados"] / len(prices)).map(lambda x: f"{x:.1%}")
    st.dataframe(coverage, use_container_width=True)
    st.download_button(
        "Baixar preços em CSV",
        prices.to_csv().encode("utf-8"),
        file_name=f"finor_{market_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
