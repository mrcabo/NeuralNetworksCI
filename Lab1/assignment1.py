import numpy as np
import matplotlib.pyplot as plt


P = 10  # number of data points
N = 4  # number of dimensions of dataset

mu, variance = 0, 1
sigma = np.sqrt(variance)
xi_mu, S_mu = np.zeros([P, N]), np.zeros([P, N], int)

for i in range(P):
    labels = np.random.binomial(1, 0.5, N)  # number of trials, probability of each trial
    # result of flipping a coin, tested N times.
    labels[labels < 1] = -1
    S_mu[i] = labels
    xi_mu[i] = np.random.normal(mu, sigma, N)


print(xi_mu, '\n', S_mu)
