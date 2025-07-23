import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# on avaerage 252 trading days in a year.
NUM_TRADING_DAYS = 252
# we will generate random w weights (different portfolios)
NUM_PORTFOLIOS = 10000

stocks = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "GOOG"]

# start_date = '2012-01-01'
# end_date = '2017-01-01'

start_date = "2015-01-01"
end_date = "2025-07-02"


def download_data():
    # name of stock as key and stocks values as the values (between dates)

    stock_data = {}

    for stock in stocks:
        # Closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

    return pd.DataFrame(stock_data)


def show_data(data):
    data.plot(figsize=(10, 8))
    plt.show()


def calculate_return(data):
    # NORMALISATION - to be able to compare variables in a comparable metric
    log_return = np.log(data / data.shift(1))  # S(t) / S(t-1)
    return log_return[1:]


def show_statistics(returns):
    # INstead of daily metric we are after annual metrics
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    # After annual return
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )

    print("Expected portfolio mean (return): %.2f" % portfolio_return)
    print("Expected portfolio volatility (std): %.2f" % portfolio_volatility)


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for i in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(
            np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w)))
        )

    return (
        np.array(portfolio_weights),
        np.array(portfolio_means),
        np.array(portfolio_risks),
    )


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker="o")
    plt.grid()
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )

    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            portfolio_return / portfolio_volatility,
        ]
    )


# scipy optimize function can find the minimum of a given function
# -max = min
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):
    # constrains the sum of the weights to 1
    contraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    # The weights can be 1 at most, 1 when 100% of the money is invested into a single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimize.minimize(
        fun=min_function_sharpe,
        x0=weights[0],
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=contraints,
    )


def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum["x"].round(3))
    print(
        "Expected return, volatility and Sharpe ratio: ",
        statistics(optimum["x"].round(3), returns),
    )


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker="o"
    )
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(
        statistics(opt["x"], rets)[1],
        statistics(opt["x"], rets)[0],
        "g*",
        markersize=20.0,
    )
    plt.show()


data_set = download_data()
# show_data(data_set)
log_daily_returns = calculate_return(data_set)
# show_statistics(log_daily_returns)

pweights, means, risks = generate_portfolios(log_daily_returns)

# show_portfolios(means, risks)

optimum = optimize_portfolio(pweights, log_daily_returns)
print_optimal_portfolio(optimum, log_daily_returns)
show_optimal_portfolio(optimum, log_daily_returns, means, risks)
