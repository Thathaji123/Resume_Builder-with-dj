import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as pe
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Stock Predictor", layout="wide")

# App title with a stylish design
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sai's Stock Dashboard</h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("<h3 style='color: #2E7D32;'>Please provide the following</h3>", unsafe_allow_html=True)
ticket_symbol = st.sidebar.text_input("Enter Ticker symbol (e.g., AAPL, GOOG)", "AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
prediction_button = st.sidebar.button("Predict Future Prices")

# Function to load data from yfinance
@st.cache_data
def load_data(stock_ticker, start_date, end_date):
    stock = yf.Ticker(stock_ticker)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Main dashboard with tabs for better organization
st.markdown(f"<h2>{ticket_symbol} Stock Overview</h2>", unsafe_allow_html=True)

# Fetch the historical stock data
if start_date and end_date:
    historical_data = load_data(ticket_symbol, start_date, end_date)

    # Display current stock price
    current_stock_price = historical_data.iloc[-1]['Open']  # Fetching the most recent stock price
    st.write(f"**Current Stock Price for {ticket_symbol}**: ${current_stock_price:.2f}")

    # Use tabs for Price Summary, Historical Data, and Charts
    price_tab, hist_tab, chart_tab = st.tabs(["📈 Price Summary", "📊 Historical Data", "📉 Charts"])

    with price_tab:
        st.write("### Price Summary")
        st.dataframe(historical_data)

    with hist_tab:
        st.write("### Historical Data")
        st.dataframe(historical_data)

    with chart_tab:
        st.write("### Stock Price Chart")
        line_chart = pe.line(historical_data, x='Date', y='Close', title=f'{ticket_symbol} Price Trend')  # Use 'Close' instead of 'Adj Close'
        st.plotly_chart(line_chart)

# Prediction model - triggered by the 'Predict Future Prices' button
if prediction_button:
    st.write(f"### Predicting future prices for {ticket_symbol}")

    def predict_stock_price(data, future_days=10):
        X = data['Open'].values.reshape(-1, 1)
        y = np.roll(X, -1)[:-1]
        X = X[:-1]

        sc = MinMaxScaler()
        X_scaled = sc.fit_transform(X)

        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_scaled, y)

        last_value_scaled = X_scaled[-1].reshape(1, -1)
        predicted_price_scaled = regressor.predict(last_value_scaled)
        predicted_next_day_price = sc.inverse_transform(predicted_price_scaled.reshape(-1, 1))

        predicted_prices = []
        current_value_scaled = last_value_scaled

        for i in range(future_days):
            predicted_price_scaled = regressor.predict(current_value_scaled)
            predicted_price = sc.inverse_transform(predicted_price_scaled.reshape(-1, 1))
            predicted_prices.append(predicted_price[0][0])
            current_value_scaled = sc.transform(predicted_price.reshape(-1, 1))

        return predicted_next_day_price[0][0], predicted_prices

    # Predict next day and next 10 days stock prices
    next_day_price, next_10_days_prices = predict_stock_price(historical_data)

    # Display results
    st.write(f"**Predicted Stock Price for Next Day**: ${next_day_price:.2f}")
    st.write("**Predicted Stock Prices for the Next 10 Days**:")
    
    # Display predicted prices in a table
    prediction_df = pd.DataFrame({
        "Day": np.arange(1, 11),
        "Predicted Price": next_10_days_prices
    })
    st.table(prediction_df)

    # Display prediction plot using Matplotlib
    st.write("### Stock Price Prediction Plot")
    days = np.arange(1, 11)
    plt.figure(figsize=(10, 5))
    plt.plot(days, next_10_days_prices, marker='o', color='blue', label='Predicted Prices')
    plt.xlabel("Day")
    plt.ylabel("Price ($)")
    plt.title(f"Predicted Stock Prices for {ticket_symbol} (Next 10 Days)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
