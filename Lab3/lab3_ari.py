import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient(w, sigma_xi, tau, example):
    gradient = (sigma_xi-tau) * (1-np.tanh(np.dot(w,example))**2) * np.sum(example)
    return gradient

dataPath = './Data/'

data3_xi = np.transpose(np.array(pd.read_csv(dataPath+'data3_xi.csv', header=None)))
data3_tau = np.transpose(np.array(pd.read_csv(dataPath+'data3_tau.csv', header=None)))

# P = len(data3_xi)
P = 100  # number of train examples
Q = 100  # number of test examples (should be equal to P, or it will blow up as we have it now..)
N = np.size(data3_xi, 1)

# learning_rate = 0.05
learning_rate = 0.05
t_max = 500  # we have to check which number we put here

# initialize weights as independent random vectors (we initialize the weights as unit vectors)
mu, variance = 0, 1
sigma = np.sqrt(variance)

# randomly initialize weights and normalize them
w1 = np.random.normal(mu, sigma, N)
w2 = np.random.normal(mu, sigma, N)
w1 = w1/np.linalg.norm(w1)
w2 = w2/np.linalg.norm(w2)

E_train = []
E_test = []

for t in range(t_max): #these are the epochs
    E_epoch_train = 0.0
    E_epoch_test = 0.0
    indexes = np.random.randint(len(data3_xi), size=(P+Q, 1))
    train_idx = indexes[np.arange(P)]
    test_idx = indexes[np.arange(P, P+Q)]
    for i, idx in enumerate(train_idx):
        example = data3_xi[idx][0]  # training example
        sigma_xi = (np.tanh(np.dot(w1, example)) + np.tanh(np.dot(w2, example)))
        e_nu = 0.5*((sigma_xi - data3_tau[idx])**2)
        E_epoch_train += e_nu[0][0]

        example_test = data3_xi[test_idx[i]][0]  # training example
        sigma_xi_test = (np.tanh(np.dot(w1, example_test)) + np.tanh(np.dot(w2, example_test)))
        e_nu_test = 0.5*((sigma_xi - data3_tau[test_idx[i]])**2)
        E_epoch_test += e_nu_test[0][0]

        #recalculate the weights
        w1 = w1 - learning_rate * gradient(w1, sigma_xi, data3_tau[idx], example)
        w2 = w2 - learning_rate * gradient(w2, sigma_xi, data3_tau[idx], example)

    E_train.append(E_epoch_train)
    E_test.append(E_epoch_test)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(np.divide(E_train, P))
ax.plot(np.divide(E_test, Q))
plt.show()
