import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date, auto_adjust=False)
    data['Price'] = ticker['Adj Close'][stock]
    return pd.DataFrame(data)


def calculate_log_returns(df):
    df['Log Return'] = np.log(df['Price'] / df['Price'].shift(1))
    return df.dropna()


def show_plot(returns, stock_name):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 6))

    # Histogram
    count, bins, _ = plt.hist(returns, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')

    # Normal distribution fit
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2, label='Normal PDF')

    plt.title(f'Daily Log Returns Distribution: {stock_name}', fontsize=14)
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stock_symbol = 'AAPL'
    stock_df = download_data(stock_symbol, '2010-01-01', '2025-07-07')
    log_returns_df = calculate_log_returns(stock_df)
    show_plot(log_returns_df['Log Return'], stock_symbol)