import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker["Close"][stock]
    return pd.DataFrame(data)


# this is how we calculate the VaR tomorrow (n=1)
def calculate_var(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1 - c))
    return var


# this is how we calculate the VaR for any days in the future
def calculate_var_n(position, c, mu, sigma, n):
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return var


if __name__ == "__main__":
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)

    stock_name = "C"

    stock_data = download_data(stock_name, start, end)

    stock_data["returns"] = np.log(
        stock_data[stock_name] / stock_data[stock_name].shift(1)
    )
    stock_data = stock_data[1:]
    print(stock_data)

    # this is the investment (stocks or whatever)
    S = 1e6
    # confidence level - this time it is 95%
    c = 0.99

    # we assume that daily returns are normally distributed
    mu = np.mean(stock_data["returns"])
    sigma = np.std(stock_data["returns"])

    print("Value at risk is: $%0.2f" % calculate_var_n(S, c, mu, sigma, 1))
