# ===============================
# 📦 IMPORTS
# ===============================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta
from xgboost import XGBRegressor

# ===============================
# ⚙️ CONFIG
# ===============================
st.set_page_config(page_title="AI Stock App", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")

# ===============================
# 📊 COMPANY LIST
# ===============================
COMPANIES = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Meta": "META",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

# ===============================
# 📌 SIDEBAR
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 Prediction", "📉 Comparison"]
)

company = st.sidebar.selectbox("Select Company", list(COMPANIES.keys()))
ticker = COMPANIES[company]

show_candle = st.sidebar.checkbox("Show Candlestick Chart")

# ===============================
# 📥 DATA
# ===============================
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    return yf.Ticker(ticker).history(period="max")

# ===============================
# 🧠 FEATURES
# ===============================
def add_features(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])

    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Lag3"] = df["Close"].shift(3)

    df["Target"] = df["Close"].pct_change().shift(-1)

    return df.dropna()

# ===============================
# 🤖 MODEL
# ===============================
FEATURES = ["Close","Volume","MA20","MA50","RSI","MACD","Lag1","Lag2","Lag3"]

def train_model(df):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(df[FEATURES], df["Target"])
    return model

def predict_next(model, last_row):
    pred_return = model.predict(last_row[FEATURES].values.reshape(1, -1))[0]
    return last_row["Close"] * (1 + pred_return)

# ===============================
# 📊 DASHBOARD
# ===============================
if page == "📊 Dashboard":

    st.header(f"{company} Overview")

    df = fetch_data(ticker)

    # SIMPLE PRICE TREND (MOST IMPORTANT)
    st.subheader("📈 Price Trend")
    st.line_chart(df["Close"])

    # OPTIONAL CANDLE
    if show_candle:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )])
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Latest Price", f"${df['Close'].iloc[-1]:.2f}")

# ===============================
# 🤖 PREDICTION PAGE
# ===============================
elif page == "🤖 Prediction":

    st.header(f"{company} Next-Day Prediction")

    df = add_features(fetch_data(ticker))

    if st.button("🔮 Predict Next Day Price"):

        model = train_model(df)
        last_row = df.iloc[-1]

        prediction = predict_next(model, last_row)

        st.success("Prediction Generated!")

        st.metric(
            "Predicted Price",
            f"${prediction:.2f}"
        )

        st.caption("⚠️ AI-based estimation (not guaranteed)")

# ===============================
# 📉 COMPARISON
# ===============================
elif page == "📉 Comparison":

    st.header("Stock Comparison")

    selected = st.multiselect(
        "Select Stocks",
        list(COMPANIES.values()),
        default=["AAPL", "TSLA", "RELIANCE.NS"]
    )

    fig = go.Figure()

    for t in selected:
        df = fetch_data(t)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            name=t
        ))

    st.plotly_chart(fig, use_container_width=True)
