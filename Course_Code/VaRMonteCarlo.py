import numpy as np
import yfinance as yf
import datetime
import pandas as pd


def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data["Close"] = ticker["Close"][stock]
    return pd.DataFrame(data)


class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price
        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.n) * rand
        )

        # we have to sort the stock prices to determine the percentile
        stock_price = np.sort(stock_price)

        # it depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


if __name__ == "__main__":
    S = 1e6  # this is the investment (stocks or whatever)
    c = 0.99  # condifence level: this time it is 99%
    n = 1  # 1 day
    iterations = 100000  # number of paths in the Monte-Carlo simulation

    # historical data to approximate mean and standard deviation
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2017, 10, 15)

    # download stock related data from Yahoo Finance
    citi = download_data("C", start_date, end_date)

    # we can use pct_change() to calculate daily returns
    citi["returns"] = citi["Close"].pct_change()

    # we can assume daily returns to be normally sidtributed: mean and variance (standard deviation)
    # can describe the process
    mu = np.mean(citi["returns"])
    sigma = np.std(citi["returns"])

    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)

    print("Value at risk with Monte-Carlo simulation: $%0.2f" % model.simulation())
