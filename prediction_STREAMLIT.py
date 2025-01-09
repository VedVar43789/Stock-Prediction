'''# Import required libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Sidebar inputs for user
st.sidebar.header("Simulation Parameters")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "SNOW")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2014-10-25"))
sample_size = st.sidebar.slider("Number of Days to Simulate", 50, 500, 100)
iterations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000)

# Fetch data from Yahoo Finance
st.title("Monte Carlo Simulation for Stock Prices")
ticker = yf.Ticker(ticker_symbol)
try:
    dataset = ticker.history(start=start_date)
    Close = dataset[['Close']]
    company_name = ticker.info.get('longName', ticker_symbol)

    # Plot historical prices
    st.subheader(f"Historical Closing Prices for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Close.index, Close['Close'], label='Closing Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Historic Price from {start_date} for {company_name}')
    ax.legend()
    st.pyplot(fig)

    # Calculate log returns
    log_return = np.log(Close['Close'] / Close['Close'].shift(1))

    # Plot historical volatility
    st.subheader(f"Log Returns (Volatility) for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(log_return.index, log_return, label='Log Returns', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Historic Volatility for {company_name}')
    ax.legend()
    st.pyplot(fig)

    # Calculate drift and deviation for simulation
    u = log_return.mean()
    var = log_return.var()
    drift = u - (0.5 * var)
    deviation = log_return.std()

    # Monte Carlo Simulation
    price_list = np.zeros((sample_size, iterations))
    price_list[0] = Close.iloc[-1]

    for t in range(1, sample_size):
        random_shock = norm.ppf(np.random.rand(iterations))
        price_list[t] = price_list[t - 1] * np.exp(drift + deviation * random_shock)

    # Plot Monte Carlo simulations
    st.subheader(f"Monte Carlo Simulations for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(iterations):
        ax.plot(price_list[:, i], lw=0.5, alpha=0.6)
    ax.set_xlabel('Days')
    ax.set_ylabel('Simulated Price')
    ax.set_title(f'Monte Carlo Simulations with Brownian Motion for {company_name}')
    st.pyplot(fig)

    # Confidence Interval
    end_prices = price_list[-1, :]
    ci_lower = np.percentile(end_prices, 45)
    ci_upper = np.percentile(end_prices, 55)
    st.write(f"Confidence Interval for the stock price after {sample_size} days: "
             f"(${ci_lower:.2f}, ${ci_upper:.2f})")

except Exception as e:
    st.error(f"An error occurred: {e}")
'''
# Import required libraries
import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

# Configure Streamlit page
st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Sidebar inputs for user
st.sidebar.header("Simulation Parameters")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "SNOW")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2014-10-25"))
sample_size = st.sidebar.slider("Number of Days to Simulate", 50, 500, 100)
iterations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000)
ci_level = st.sidebar.slider("Confidence Interval (%)", 90, 99, 95)

# Fetch data from Yahoo Finance
st.title("Monte Carlo Simulation for Stock Prices")
ticker = yf.Ticker(ticker_symbol)

try:
    # Fetch historical data
    dataset = ticker.history(start=start_date)
    Close = dataset[['Close']]
    company_name = ticker.info.get('longName', ticker_symbol)

    # Plot historical prices
    st.subheader(f"Historical Closing Prices for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Close.index, Close['Close'], label='Closing Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Historic Price from {start_date} for {company_name}')
    ax.legend()
    st.pyplot(fig)

    # Calculate log returns
    log_return = np.log(Close['Close'] / Close['Close'].shift(1))

    # Plot historical volatility
    st.subheader(f"Log Returns (Volatility) for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(log_return.index, log_return, label='Log Returns', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Historic Volatility for {company_name}')
    ax.legend()
    st.pyplot(fig)

    # Calculate drift and deviation for simulation
    u = log_return.mean()
    var = log_return.var()
    drift = u - (0.5 * var)
    deviation = log_return.std()

    # Monte Carlo Simulation
    price_list = np.zeros((sample_size, iterations))
    price_list[0] = Close.iloc[-1]

    for t in range(1, sample_size):
        random_shock = norm.ppf(np.random.rand(iterations))
        price_list[t] = price_list[t - 1] * np.exp(drift + deviation * random_shock)

    # Plot Monte Carlo simulations
    st.subheader(f"Monte Carlo Simulations for {company_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(iterations):
        ax.plot(price_list[:, i], lw=0.5, alpha=0.6)
    ax.set_xlabel('Days')
    ax.set_ylabel('Simulated Price')
    ax.set_title(f'Monte Carlo Simulations with Brownian Motion for {company_name}')
    st.pyplot(fig)

    # Confidence Interval
    end_prices = price_list[-1, :]
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower = np.percentile(end_prices, lower_percentile)
    ci_upper = np.percentile(end_prices, upper_percentile)

    st.write(f"{ci_level}% Confidence Interval for the stock price after {sample_size} days: "
             f"(${ci_lower:.2f}, ${ci_upper:.2f})")

except Exception as e:
    st.error(f"An error occurred: {e}")
