import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import matplotlib.pyplot as plt

# Function to load stock data using yfinance
@st.cache_data
def load_data(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    data = stock.history(period="5y")  # Fetch the last 5 years of data
    data.reset_index(inplace=True)
    return data

# Function to preprocess data and make predictions
def predict_stock_price(data, future_days=10):
    # Extract the 'Open' column for the features (X)
    X = data['Open'].values.reshape(-1, 1)

    # Create the target variable (y), which is the 'Open' price shifted by 1 day
    y = np.roll(X, -1)[:-1]
    X = X[:-1]

    # Scale the features between 0 and 1
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    # Train the Random Forest Regressor on the entire dataset
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_scaled, y)

    # Get the last data point (most recent stock price)
    last_value_scaled = X_scaled[-1].reshape(1, -1)

    # Predict the next day's stock price
    predicted_price_scaled = regressor.predict(last_value_scaled)
    predicted_next_day_price = sc.inverse_transform(predicted_price_scaled.reshape(-1, 1))

    # Predict the next 10 days of stock prices
    predicted_prices = []
    current_value_scaled = last_value_scaled

    for i in range(future_days):
        predicted_price_scaled = regressor.predict(current_value_scaled)
        predicted_price = sc.inverse_transform(predicted_price_scaled.reshape(-1, 1))
        predicted_prices.append(predicted_price[0][0])
        current_value_scaled = sc.transform(predicted_price.reshape(-1, 1))

    return predicted_next_day_price[0][0], predicted_prices

# Streamlit App
st.title('Stock Price Predictor')

# User input to select stock ticker
stock_ticker = st.text_input("Enter the Stock Ticker (e.g., GOOG for Google)", "GOOG")

# Button to predict stock prices
if st.button("Predict"):
    # Load data for the selected stock ticker
    stock_data = load_data(stock_ticker)

    # Display the current stock price (latest available price)
    current_stock_price = stock_data.iloc[-1]['Open']
    st.write(f"**Current Stock Price for {stock_ticker}**: {current_stock_price}")

    # Predict next day and next 10 days stock prices
    next_day_price, next_10_days_prices = predict_stock_price(stock_data)

    # Display the next day's predicted stock price
    st.write(f"**Predicted Stock Price for Next Day**: {next_day_price}")

    # Display the predicted stock prices for the next 10 days
    st.write(f"**Predicted Stock Prices for the Next 10 Days**:")
    for i, price in enumerate(next_10_days_prices, start=1):
        st.write(f"Day {i}: {price}")

    # Plot the results using Matplotlib
    st.write("### Stock Price Prediction Plot")
    days = np.arange(1, 11)
    plt.figure(figsize=(10, 5))
    plt.plot(days, next_10_days_prices, label="Predicted Prices", marker='o')
    plt.xlabel("Day")
    plt.ylabel("Predicted Stock Price")
    plt.title(f"Predicted Stock Prices for {stock_ticker} (Next 10 Days)")
    plt.grid(True)
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
