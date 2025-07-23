import matplotlib.pyplot as plt
import numpy as np

"""
Simulates and plots a Geometric Brownian Motion (GBM), a common model for stock prices.

This script includes:
- A function to simulate a GBM path using the stochastic differential equation:
    S(t) = S0 * exp[(mu - 0.5 * sigma^2) * t + sigma * W(t)]
  where:
    S0     = initial stock price
    mu     = expected return (drift)
    sigma  = volatility
    W(t)   = Wiener process (standard Brownian motion)
    T      = total time
    N      = number of time steps

- A function to plot the simulated stock price over time.

Usage:
Run the script directly to simulate a single GBM path with default parameters and visualise it.

Author: Wadoud
"""


def simulate_geometric_random_walk(S0, T=2, N=1000, mu=0.1, sigma=0.05):
    dt = T / N
    t = np.linspace(0, T, N)
    # standard normal distribution N(0,1)
    W = np.random.standard_normal(size=N)
    # N(0,dt) = sqrt(dt) * N(0,1)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)

    return t, S


def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price S(t)")
    plt.title("Geometric Brownian Motion")
    plt.show()


if __name__ == "__main__":
    time, data = simulate_geometric_random_walk(10)
    plot_simulation(time, data)
