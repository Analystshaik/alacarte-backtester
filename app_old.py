from __future__ import annotations
import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="À-La-Carte Stock & Options Backtester", layout="wide")

DB = Path("data") / "streamlit_data.sqlite"

SPY = [
"AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK.B","AVGO","TSLA","JPM","LLY","V",
"XOM","UNH","MA","COST","WMT","NFLX","JNJ","HD","PG","ORCL","ABBV","BAC","KO","CRM","CVX",
"MRK","AMD","PEP","CSCO","TMO","ACN","MCD","ABT","LIN","WFC","DHR","GE","INTU","IBM","QCOM",
"DIS","PM","TXN","CAT","AMGN","RTX","VZ","NOW","PFE","AMAT","ISRG","AXP","SPGI","NEE","BKNG",
"LOW","COP","HON","GS","UBER","SYK","BLK","ELV","TJX","VRTX","ADI","MDT","PANW","DE","LMT",
"REGN","ADP","CB","ETN","BMY","MMC","SBUX","GILD","PLD","C","CI","MU","FI","BA","SO","KLAC",
"DUK","SHW","ICE","MCK","SNPS","CDNS","MO","APH","WM","EQIX","NOC","CL","UPS","MDLZ","USB",
"CMG","PYPL","MSI","GD","EOG","EMR","TDG","PNC","CEG","MCO","ORLY","ITW","CTAS","MAR","AON",
"CSX","HCA","TGT","FCX","BDX","CME","NXPI","SLB","NSC","ROP","AJG","FDX","PSA","COF","APD",
"ADSK","EW","NEM","TRV","HLT","O","PCAR","AEP","AZO","MET","CPRT","AFL","F","GM","WELL","MNST",
"MSCI","D","BK","KMB","OKE","RSG","ROST","PSX","AMP","ALL","LHX","TEL","VLO","SRE","GWW","KMI",
"URI","FAST","PAYX","DOW","AME","OXY","COR","YUM","FIS","HUM","A","LEN","IQV","DHI","CTVA",
"EA","EXC","PRU","OTIS","KR","IDXX","ACGL","KDP","ODFL","FANG","CMI","CTSH","HSY","GIS","MLM",
"GEHC","PEG","IR","VRSK","ED","RCL","KVUE","XEL","PCG","DD","HWM","EXR","VMC","NDAQ","CBRE"
]

QQQ = [
"AAPL","MSFT","NVDA","AMZN","META","AVGO","GOOGL","GOOG","TSLA","COST","NFLX","TMUS","AMD",
"PEP","CSCO","LIN","ADBE","QCOM","TXN","INTU","AMGN","AMAT","ISRG","BKNG","HON","VRTX","ADI",
"PANW","ADP","SBUX","GILD","MU","KLAC","MDLZ","PYPL","SNPS","CDNS","MAR","CSX","MELI","ORLY",
"PCAR","CPRT","MNST","ROP","ROST","FAST","PAYX","EA","KDP","ODFL","IDXX","CTSH","VRSK","XEL",
"DXCM","TEAM","ZS","DDOG","CRWD","MRVL","FTNT","LRCX","NXPI","MCHP","ON","TTD","WDAY","CEG",
"EXC","AEP","GEHC","AZN","BIIB","REGN","ILMN","WBD","CHTR","CCEP","CSGP","ANSS","MDB","ARM",
"APP","PLTR","ABNB","GFS","SMCI","CDW","TTWO","FANG","BKR","DLTR","PDD","JD","BIDU"
]

DIA = [
"AMGN","AXP","AMZN","AAPL","BA","CAT","CRM","CSCO","CVX","DIS","GS","HD","HON","IBM","JNJ",
"JPM","KO","MCD","MMM","MRK","MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT"
]

UNIVERSES = {"SPY": SPY, "QQQ": QQQ, "DIA": DIA}

def normalize_date_expr():
    return """
    CASE
      WHEN length(CAST(bar_date AS TEXT)) = 8
           AND instr(CAST(bar_date AS TEXT), '-') = 0
      THEN substr(CAST(bar_date AS TEXT),1,4)||'-'||
           substr(CAST(bar_date AS TEXT),5,2)||'-'||
           substr(CAST(bar_date AS TEXT),7,2)
      ELSE CAST(bar_date AS TEXT)
    END
    """

@st.cache_data(ttl=300)
def available_symbols():
    if not DB.exists():
        return []
    con = sqlite3.connect(DB)
    try:
        rows = con.execute("SELECT DISTINCT symbol FROM relationship_bars_day ORDER BY symbol").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]

@st.cache_data(ttl=300)
def symbol_date_range(symbol):
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            f"""SELECT MIN(date({normalize_date_expr()})),
                       MAX(date({normalize_date_expr()})),
                       COUNT(*)
                FROM relationship_bars_day
                WHERE symbol=?""",
            (symbol,)
        ).fetchone()
    finally:
        con.close()
    return row

@st.cache_data(ttl=120)
def load_stock(symbol, start_date, end_date):
    con = sqlite3.connect(DB)
    try:
        q = f"""
        SELECT date({normalize_date_expr()}) AS bar_date,
               open, high, low, close, volume
        FROM relationship_bars_day
        WHERE symbol=?
          AND date({normalize_date_expr()}) >= date(?)
          AND date({normalize_date_expr()}) <= date(?)
        ORDER BY date({normalize_date_expr()})
        """
        df = pd.read_sql_query(q, con, params=(symbol, str(start_date), str(end_date)))
    finally:
        con.close()

    if df.empty:
        return df

    df["bar_date"] = pd.to_datetime(df["bar_date"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["bar_date","open","high","low","close"])

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, r, sigma, option_type="CALL", q=0.0):
    if T <= 0:
        return max(S-K,0.0) if option_type=="CALL" else max(K-S,0.0)
    if sigma <= 0:
        sigma = 1e-9
    d1 = (math.log(S/K) + (r-q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type=="CALL":
        return S*math.exp(-q*T)*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)
    return K*math.exp(-r*T)*norm_cdf(-d2) - S*math.exp(-q*T)*norm_cdf(-d1)

def strike_step(price):
    if price < 25: return 1.0
    if price < 100: return 2.5
    if price < 250: return 5.0
    return 10.0

def atm_strike(price):
    step = strike_step(price)
    return round(price / step) * step

def third_friday(year, month):
    d = date(year, month, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d

def expiry_at_least(entry_date, min_days=182):
    target = entry_date + timedelta(days=min_days)
    y, m = target.year, target.month
    exp = third_friday(y, m)
    if exp < target:
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
        exp = third_friday(y, m)
    return exp

def stock_backtest(df, direction, target_pct, stop_pct, use_stop, max_days):
    if df.empty:
        return None

    entry = df.iloc[0]
    entry_px = float(entry["open"])
    entry_date = entry["bar_date"]

    if direction == "LONG":
        target = entry_px * (1 + target_pct/100)
        stop = entry_px * (1 - stop_pct/100)
    else:
        target = entry_px * (1 - target_pct/100)
        stop = entry_px * (1 + stop_pct/100)

    exit_px = float(df.iloc[-1]["close"])
    exit_date = df.iloc[-1]["bar_date"]
    reason = "END_DATE"

    for i, row in df.iterrows():
        days = (row["bar_date"] - entry_date).days
        if days > max_days:
            exit_px = float(row["close"])
            exit_date = row["bar_date"]
            reason = "MAX_HOLD"
            break

        if direction == "LONG":
            if row["high"] >= target:
                exit_px, exit_date, reason = target, row["bar_date"], "TARGET"
                break
            if use_stop and row["low"] <= stop:
                exit_px, exit_date, reason = stop, row["bar_date"], "STOP"
                break
        else:
            if row["low"] <= target:
                exit_px, exit_date, reason = target, row["bar_date"], "TARGET"
                break
            if use_stop and row["high"] >= stop:
                exit_px, exit_date, reason = stop, row["bar_date"], "STOP"
                break

    pnl_pct = ((exit_px-entry_px)/entry_px*100) * (1 if direction=="LONG" else -1)

    return {
        "entry_date": entry_date.date(),
        "entry_price": entry_px,
        "exit_date": exit_date.date(),
        "exit_price": exit_px,
        "exit_reason": reason,
        "return_pct": pnl_pct
    }

def option_backtest(df, option_type, target_pct, iv, risk_free, dividend_yield, max_days):
    if df.empty:
        return None, pd.DataFrame()

    entry = df.iloc[0]
    entry_date = entry["bar_date"].date()
    S0 = float(entry["open"])
    K = atm_strike(S0)
    expiry = expiry_at_least(entry_date, 182)
    T0 = max((expiry-entry_date).days/365.0, 1/365)
    p0 = bs_price(S0, K, T0, risk_free, iv, option_type, dividend_yield)

    target = p0*(1+target_pct/100)
    exit_price = p0
    exit_date = entry_date
    reason = "END_DATE"
    path=[]

    for _, row in df.iterrows():
        d = row["bar_date"].date()
        held = (d-entry_date).days
        if held > max_days:
            S = float(row["close"])
            T = max((expiry-d).days/365.0, 0)
            exit_price = bs_price(S,K,T,risk_free,iv,option_type,dividend_yield)
            exit_date=d; reason="MAX_HOLD"
            break

        if d >= expiry:
            S = float(row["close"])
            exit_price = max(S-K,0) if option_type=="CALL" else max(K-S,0)
            exit_date=d; reason="EXPIRY"
            path.append((d,S,exit_price))
            break

        # Conservative-ish path approximation using favorable intraday extreme for target test.
        S_test = float(row["high"] if option_type=="CALL" else row["low"])
        T = max((expiry-d).days/365.0, 1/365)
        opt_test = bs_price(S_test,K,T,risk_free,iv,option_type,dividend_yield)
        path.append((d,S_test,opt_test))

        if opt_test >= target:
            exit_price=target; exit_date=d; reason="TARGET"
            break

        S_close=float(row["close"])
        exit_price=bs_price(S_close,K,T,risk_free,iv,option_type,dividend_yield)
        exit_date=d

    pnl_pct=(exit_price-p0)/p0*100 if p0 else np.nan
    result={
        "entry_date":entry_date,
        "underlying_entry":S0,
        "strike":K,
        "expiry":expiry,
        "option_type":option_type,
        "entry_option_price":p0,
        "exit_date":exit_date,
        "exit_option_price":exit_price,
        "exit_reason":reason,
        "return_pct":pnl_pct
    }
    path_df=pd.DataFrame(path,columns=["date","underlying_test_price","model_option_price"])
    return result,path_df

st.title("À-La-Carte Stock & Options Backtester")
st.caption(
    "No signal logic: choose the stock, dates and trade directly. "
    "Stock prices come from the cloud SQLite database. Option prices are modeled with Black-Scholes; "
    "they are not historical option-chain fills."
)

if not DB.exists():
    st.error("Cloud price database not found: data/streamlit_data.sqlite")
    st.stop()

all_loaded = set(available_symbols())

with st.sidebar:
    st.header("Trade Setup")
    universe_name = st.selectbox("Universe", ["SPY","QQQ","DIA"])
    universe_symbols = [s for s in UNIVERSES[universe_name] if s in all_loaded]

    if not universe_symbols:
        st.error(f"No {universe_name} symbols are available in the cloud database.")
        st.stop()

    symbol = st.selectbox("Stock", universe_symbols)
    coverage = symbol_date_range(symbol)
    min_d = pd.to_datetime(coverage[0]).date() if coverage and coverage[0] else date.today()-timedelta(days=365)
    max_d = pd.to_datetime(coverage[1]).date() if coverage and coverage[1] else date.today()

    st.caption(f"DB coverage: {min_d} to {max_d} ({coverage[2]:,} rows)")

    start_date = st.date_input("Start date", value=max(min_d, max_d-timedelta(days=365)), min_value=min_d, max_value=max_d)
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)

    mode = st.radio("Instrument", ["Stock","Option"])

    st.divider()

    target_pct = st.number_input("Profit target %", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    max_hold = st.number_input("Maximum holding days", min_value=1, max_value=730, value=180, step=1)

    if mode=="Stock":
        stock_direction = st.radio("Stock direction", ["LONG","SHORT"], horizontal=True)
        use_stop = st.checkbox("Use stop loss", value=False)
        stop_pct = st.number_input("Stop loss %", min_value=1.0, max_value=100.0, value=10.0, step=1.0, disabled=not use_stop)
    else:
        option_type = st.radio("Option type", ["CALL","PUT"], horizontal=True)
        iv_pct = st.number_input("Implied volatility %", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
        risk_free_pct = st.number_input("Risk-free rate %", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
        dividend_yield_pct = st.number_input("Dividend yield %", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    run = st.button("Run Backtest", type="primary", use_container_width=True)

if run:
    if end_date < start_date:
        st.error("End date must be on or after start date.")
        st.stop()

    df = load_stock(symbol,start_date,end_date)
    if df.empty:
        st.error(f"No price data found for {symbol} between {start_date} and {end_date}.")
        st.stop()

    st.subheader(f"{symbol} Backtest")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("First available bar", df.iloc[0]["bar_date"].date())
    c2.metric("Last available bar", df.iloc[-1]["bar_date"].date())
    c3.metric("Bars", len(df))
    c4.metric("Instrument", mode)

    if mode=="Stock":
        res=stock_backtest(df,stock_direction,target_pct,stop_pct,use_stop,max_hold)
        cols=st.columns(5)
        cols[0].metric("Entry", f"${res['entry_price']:.2f}")
        cols[1].metric("Exit", f"${res['exit_price']:.2f}")
        cols[2].metric("Return", f"{res['return_pct']:.2f}%")
        cols[3].metric("Exit reason", res["exit_reason"])
        cols[4].metric("Exit date", str(res["exit_date"]))

        st.dataframe(pd.DataFrame([res]),use_container_width=True,hide_index=True)

    else:
        res,path=option_backtest(
            df,option_type,target_pct,iv_pct/100.0,risk_free_pct/100.0,
            dividend_yield_pct/100.0,max_hold
        )
        cols=st.columns(6)
        cols[0].metric("Strike", f"${res['strike']:.2f}")
        cols[1].metric("Expiry", str(res["expiry"]))
        cols[2].metric("Entry option", f"${res['entry_option_price']:.2f}")
        cols[3].metric("Exit option", f"${res['exit_option_price']:.2f}")
        cols[4].metric("Return", f"{res['return_pct']:.2f}%")
        cols[5].metric("Exit reason", res["exit_reason"])

        st.dataframe(pd.DataFrame([res]),use_container_width=True,hide_index=True)

        if not path.empty:
            chart=path.set_index("date")[["model_option_price"]]
            st.line_chart(chart)

    price_chart=df.set_index("bar_date")[["close"]]
    st.subheader(f"{symbol} Close Price")
    st.line_chart(price_chart)

st.divider()
st.caption(
    "Option backtests are model-based. The app selects an ATM-ish strike and the nearest standard "
    "monthly third-Friday expiration at least 182 days after entry, then values the option with "
    "Black-Scholes using your IV/rate/dividend inputs."
)
