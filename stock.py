import streamlit as st
import pandas as pd
import yfinance as yf

import plotly.express as pe
st.title("Sai's Stock Dashboard")
st.sidebar.title('please provideing the following')

ticket_symbol=st.sidebar.text_input("Enter Ticker symbol", "AAPL")
start_data=st.sidebar.date_input("Start Date", value=None)
end_date=st.sidebar.date_input("End Date", value=None)


ticker=yf.Ticker(ticket_symbol)
st.write("Welcome")
historical_data=ticker.history(start=start_data,end=end_date)
# st.write(historical_data)

if start_data is not None and end_date is not None:
    # st.write(historical_data)

    st.subheader(f'{ticket_symbol} Stock Overview')
    stockData=yf.download(ticket_symbol,start=start_data,end=end_date)
    price_tab,hist_tab,chart_tab=st.tabs(["Price Summary","Historical Data","Charts"])

    with price_tab:
        st.write("Price Summary")
        st.write(stockData)
    with hist_tab:
        st.write("Historical Data")
        st.write(historical_data)
    with chart_tab:
        st.write("Charts")
        line_charts=pe.line(stockData, stockData.index,y=stockData['Adj Close'],title=ticket_symbol)
        st.plotly_chart(line_charts)
    


