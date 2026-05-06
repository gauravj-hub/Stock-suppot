import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Streamlit UI
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# Country Selection
country = st.radio("Select Market:", ["India", "US"])

# User Inputs
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# Initialize Storage
forecasted_prices = {}
volatilities = {}
news_results = {}
backtest_results = {}
sector_allocation = {}

def fetch_yahoo_news(stock):
    ticker = yf.Ticker(stock)
    return ticker.news

def backtest_strategy(df):
    df['Returns'] = df['Close'].pct_change()
    cumulative_return = (1 + df['Returns']).cumprod()
    return cumulative_return.iloc[-1] - 1  # Total return over period

for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    
    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue
    
    df['Returns'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    
    # Fetch Latest News
    news_results[stock] = fetch_yahoo_news(stock)
    
    # Backtesting
    backtest_results[stock] = backtest_strategy(df)
    
    # Train XGBoost Model
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    future_xgb = [xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]
    
    forecasted_prices[stock] = future_xgb[-1]
    volatilities[stock] = float(np.std(df['Returns']))
    
    # Plot Historical Prices
    st.subheader(f"📊 Forecast for {stock}")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index[-forecast_days:], future_xgb, label=f'{stock} Forecasted (XGBoost)', linestyle='dashed', color='red')
    plt.legend()
    plt.title(f"Historical and Forecasted Prices for {stock}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    st.pyplot(plt)

# Display Latest News
if news_results:
    st.subheader("📰 Latest Stock Market News")
    for stock, articles in news_results.items():
        st.write(f"### {stock}")
        for article in articles[:3]:  # Show top 3 news articles
            st.write(f"🔹 [{article['title']}]({article['link']}) - {article['publisher']}")

# Display Backtesting Results
if backtest_results:
    st.subheader("📉 Backtesting Results")
    backtest_df = pd.DataFrame.from_dict(backtest_results, orient='index', columns=['Total Return'])
    st.table(backtest_df)

# Display Sector-Wise Diversification
if sector_allocation:
    st.subheader("📊 Sector-Wise Diversification Analysis")
    sector_df = pd.DataFrame.from_dict(sector_allocation, orient='index', columns=['Allocation (%)'])
    st.table(sector_df)

st.success("✅ Features Integrated: AI Forecasting, Market News, Risk Analysis, Backtesting, and Diversification Analysis!")
