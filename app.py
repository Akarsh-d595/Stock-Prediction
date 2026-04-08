import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np
import datetime

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(ticker, period='5y', interval='1d'):
    """Fetches historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the symbol.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def create_features(df):
    """Creates technical indicators as features for the model."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Volume_MA50'] = df['Volume'].rolling(window=50).mean() # Add Volume MA as feature
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['Open_Close_Diff'] = df['Open'] - df['Close']

    # Target: Next day's close price
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with NaN values created by rolling means or shift
    df = df.dropna()
    return df

def train_model(df):
    """Trains a RandomForestRegressor model and returns the model and feature columns."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200',
                'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Volume_MA50',
                'High_Low_Diff', 'Open_Close_Diff']

    X = df[features]
    y = df['Target']

    # Use the last day's features for prediction, so split before the last row
    # For training, split historical data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    st.sidebar.write(f"Model RMSE on test set: {rmse:.2f}")

    return model, features

def predict_next_day_close(model, last_day_data, features):
    """Predicts the next day's closing price."""
    # Ensure the input data has the same features as training data
    prediction_input = last_day_data[features].values.reshape(1, -1)
    next_day_price = model.predict(prediction_input)[0]
    return next_day_price

# --- Plotting Functions ---
def plot_candlestick(df, ticker):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            )
        ]
    )
    fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def plot_volume(df, ticker):
    fig = go.Figure(
        data=[
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='skyblue'
            )
        ]
    )
    fig.update_layout(
        title=f'{ticker} Volume',
        yaxis_title='Volume',
        xaxis_rangeslider_visible=False,
        height=250
    )
    return fig

def plot_comparison(df, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=predictions, mode='lines', name='Predicted Close', line=dict(color='red', dash='dot')))
    fig.update_layout(
        title=f'{ticker} Actual vs. Predicted Close Prices',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        height=500
    )
    return fig

# --- Streamlit UI ---
st.title("📈 Stock Price Prediction System")
st.markdown("Enter a stock ticker symbol to get historical data and predict the next day's closing price.")

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()

if st.button("Predict"):
    if not ticker_input:
        st.warning("Please enter a stock ticker symbol.")
    else:
        st.subheader(f"Analyzing {ticker_input}...")

        # 1. Fetch Data
        data_load_state = st.info("Fetching data...")
        df = fetch_stock_data(ticker_input)
        data_load_state.empty()

        if df is not None and not df.empty:
            st.success("Data fetched successfully!")

            # 2. Feature Engineering
            df_features = create_features(df.copy())

            if df_features.empty:
                st.warning("Not enough data after feature engineering. Please check the ticker or data period.")
            else:
                # Get the last day's features for the actual prediction
                last_day_for_prediction = df_features.iloc[-1].drop('Target') # Exclude 'Target' from features

                # For plotting and model training, we need data up to the second to last day to have 'Target' values
                df_model_ready = df_features.iloc[:-1]

                if df_model_ready.empty:
                    st.warning("Not enough data to train the model after preparing target. Please check the ticker or data period.")
                else:
                    # 3. Train Model
                    model_load_state = st.info("Training model...")
                    model, features = train_model(df_model_ready)
                    model_load_state.empty()
                    st.success("Model trained successfully!")

                    # 4. Make Prediction for next day
                    predicted_next_day_close = predict_next_day_close(model, last_day_for_prediction.to_frame().T, features)

                    st.markdown("--- ")
                    st.subheader("Prediction for Next Trading Day")
                    col_pred_metric, col_pred_info = st.columns([1, 2])
                    with col_pred_metric:
                        st.metric(label=f"Predicted Close Price for {ticker_input}", value=f"${predicted_next_day_close:.2f}", delta_color="off")
                    with col_pred_info:
                        st.write("\nThis prediction is based on historical data and technical indicators.")
                        st.write(f"*Last available data point: {df.index[-1].strftime('%Y-%m-%d')} with Close: ${df['Close'].iloc[-1]:.2f}*")
                    st.markdown("--- ")

                    # 5. Generate historical predictions for comparison plot
                    # Retrain on full data for comparison plot, or just use the test set predictions
                    # For simplicity, let's just predict on all available data (excluding the very last row used for next day's prediction)
                    X_all = df_model_ready[features]
                    historical_predictions = model.predict(X_all)
                    df_model_ready['Historical_Predicted_Close'] = historical_predictions

                    # --- Display Charts ---
                    st.subheader("Interactive Charts")

                    # Candlestick Chart
                    st.plotly_chart(plot_candlestick(df, ticker_input), use_container_width=True)

                    # Volume Bar Chart
                    st.plotly_chart(plot_volume(df, ticker_input), use_container_width=True)

                    # Comparison Plot (Actual vs. Historical Predicted)
                    st.plotly_chart(plot_comparison(df_model_ready, df_model_ready['Historical_Predicted_Close'], ticker_input), use_container_width=True)
        else:
            st.error("Could not retrieve stock data or data was empty. Please ensure the ticker is correct and try again.")

st.sidebar.markdown("### About this App")
st.sidebar.info(
    "This application predicts the next day's stock closing price using a Random Forest Regressor model. "
    "It uses `yfinance` to fetch historical data and `Plotly` for interactive visualizations."\
    "Features include 50-day and 200-day Moving Averages, volume, and date-related features."
)

st.sidebar.markdown("Developed by Akarsh Dubey/AI Assistant")
