import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title("Stock Price Forecast using ARIMA")

ticker = st.text_input("Stock Ticker", "AAPL")

if st.button("Run Forecast"):

    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True
    )

    close = df["Close"]

    st.subheader("Historical Price")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(close)
    ax.set_title(f"{ticker} - Last 5 Years")
    st.pyplot(fig)

    model = ARIMA(close, order=(5, 1, 0))
    model_fit = model.fit()

    forecast_days = 365

    forecast = model_fit.forecast(steps=forecast_days)

    future_dates = pd.date_range(
        start=close.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = pd.DataFrame(
        {"Forecast": forecast},
        index=future_dates
    )

    st.subheader("Forecast")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(close[-250:], label="Historical")
    ax2.plot(forecast_df, label="Forecast")
    ax2.legend()

    st.pyplot(fig2)

    june_2027 = forecast_df.loc[
        forecast_df.index.strftime("%Y-%m") == "2027-06"
    ]

    if not june_2027.empty:
        st.subheader("June 2027 Forecast")
        st.dataframe(june_2027)
