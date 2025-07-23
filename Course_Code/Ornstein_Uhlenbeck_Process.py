import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Ornstein-Uhlenbeck process
theta = 0.7  # rate of mean reversion
mu = 1.5  # long-term mean
sigma = 0.3  # volatility
X0 = 0.0  # initial value

# Simulation parameters
T = 50.0  # total time
N = 10000  # number of time steps
dt = T / N  # time step
t = np.linspace(0, T, N)

# Initialise process
X = np.zeros(N)
X[0] = X0

# Simulate the OU process
for i in range(1, N):
    X[i] = (
        X[i - 1]
        + theta * (mu - X[i - 1]) * dt
        + sigma * np.sqrt(dt) * np.random.randn()
    )

# Plot the process
plt.figure(figsize=(12, 5))
plt.plot(t, X, label="Ornstein-Uhlenbeck Process (Long Time)")
plt.axhline(mu, color="r", linestyle="--", label="Mean (mu)")
plt.title("Long-Term Simulation of the Ornstein-Uhlenbeck Process")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
