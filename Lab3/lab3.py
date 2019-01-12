import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


def gradient(w, sigma_xi, tau, example):
    gradient = (sigma_xi-tau) * (1-np.tanh(np.dot(w,example))**2) * np.sum(example)
    return gradient


def findE(xi, tau, w1, w2):
    P = np.size(xi, 0)
    E = 0
    for i, xi_nu in enumerate(xi):
        sigma_xi = (np.tanh(np.dot(w1, xi_nu)) + np.tanh(np.dot(w2, xi_nu)))
        E += (sigma_xi - tau[i]) ** 2
    E = E / (2*P)
    return E


dataPath = './Data/'

data3_xi = np.transpose(np.array(pd.read_csv(dataPath+'data3_xi.csv', header=None)))
data3_tau = np.transpose(np.array(pd.read_csv(dataPath+'data3_tau.csv', header=None)))

# P = len(data3_xi)
P = 200  # number of train examples
Q = 100  # number of test examples
N = np.size(data3_xi, 1)

learning_rate = 0.05
# learning_rate = 0.001
t_max = 300  # we have to check which number we put here

# initialize weights as independent random vectors (we initialize the weights as unit vectors)
mu, variance = 0, 1
sigma = np.sqrt(variance)

E_mean = np.zeros((t_max, 1))
E_test_mean = np.zeros((t_max, 1))
n_runs = 10
for i in range(n_runs):
    # randomly initialize weights and normalize them
    w1 = np.random.normal(mu, sigma, N)
    w2 = np.random.normal(mu, sigma, N)
    w1 = w1/np.linalg.norm(w1)
    w2 = w2/np.linalg.norm(w2)

    E_train = []
    E_test = []

    for t in range(t_max): #these are the epochs
        shuffled_data, shuffled_tau = shuffle(data3_xi, data3_tau)
        train_idx = np.arange(P)
        test_idx = np.arange(P, P+Q)

        for idx in train_idx:  # P sampled examples
            xi_nu = shuffled_data[idx]  # training example
            sigma_xi = (np.tanh(np.dot(w1, xi_nu)) + np.tanh(np.dot(w2, xi_nu)))
            tau_xi = shuffled_tau[idx]
            #recalculate the weights
            w1 = w1 - learning_rate * gradient(w1, sigma_xi, tau_xi, xi_nu)
            w2 = w2 - learning_rate * gradient(w2, sigma_xi, tau_xi, xi_nu)

        E_epoch_train = findE(shuffled_data[train_idx], shuffled_tau[train_idx], w1, w2)
        E_epoch_test = findE(shuffled_data[test_idx], shuffled_tau[test_idx], w1, w2)
        E_train.append(E_epoch_train)
        E_test.append(E_epoch_test)
    E_mean = np.add(E_mean, np.asarray(E_train))
    E_test_mean = np.add(E_test_mean, np.asarray(E_test))

fig = plt.figure()
plt.plot(np.divide(E_mean, n_runs))
plt.plot(np.divide(E_test_mean, n_runs))
plt.legend(["E_mean", "E_test_mean"])
plt.xlabel("t")
fig.savefig('./outputs/E_cost.png')
plt.show()

fig2 = plt.figure()
idx = np.arange(np.size(w1))

rects1 = plt.bar(idx, w1, color='b', label='w1')

rects2 = plt.bar(idx, w2, color='g', label='w2')
plt.legend()

fig2.savefig('./outputs/weights.png')
plt.show()

