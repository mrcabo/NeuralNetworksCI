import numpy as np
import matplotlib.pyplot as plt


# Run your code in order to study Perceptron training at least for the following parameter settings:
# N = 20, P = αN with α = 0.75, 1.0, 1.25, . . . 3.0, nD =50, n_max =100

alpha = 1.0
N = 20  # number of dimensions of dataset
P = round(alpha*N)  # number of data points (rounded to closest int)

mu, variance = 0, 1
sigma = np.sqrt(variance)
xi_mu, S_mu = np.zeros([P, N]), np.zeros(P, int)  # TODO: should we add the threshold theta?

for i in range(P):
    labels = np.random.binomial(1, 0.5, 1)  # number of trials, probability of each trial
    # result of flipping a coin, tested N times.
    labels[labels < 1] = -1
    S_mu[i] = labels
    xi_mu[i] = np.random.normal(mu, sigma, N)


# print(xi_mu, '\n', S_mu)


def perceptron(X, Y):
    w = np.zeros(len(X[0]))
    epochs = 100
    done = False

    for t in range(epochs):
        done = True
        for i in range(P):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + (1/N)*X[i]*Y[i]
                done = False
        if done:  # TODO: Is this what he means by  ..performed until either a solution with all Eν > 0
            break

    return w


w = perceptron(xi_mu, S_mu)

print(w)
