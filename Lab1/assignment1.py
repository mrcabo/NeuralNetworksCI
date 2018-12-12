import numpy as np
import matplotlib.pyplot as plt
import progressbar


# I put it in a function so that we can change the values for alpha easily
def experiment(N, n_max, nD, alpha):
    P = int(round(alpha*N))  # number of data points (rounded to closest int)
    mu, variance = 0, 1
    sigma = np.sqrt(variance)
    # theta = -1.0  # to make it inhomogeneous
    data = []
    # generate nD datasets
    for _ in range(nD):
        labels = np.random.binomial(1, 0.5, (P, 1))  # number of trials, probability of each trial
        # result of flipping a coin, tested N times.
        labels[labels < 1] = -1
        S_mu = labels
        xi_mu = np.random.normal(mu, sigma, (P,N))
        # theta_arr = np.full((P, 1), theta)
        # xi_mu = np.concatenate((xi_mu, theta_arr), axis=1)
        data.append([xi_mu, S_mu])
        
    success = perceptron(data, N, P, n_max)
    return success


def perceptron(data, N, P, n_max):


    success=0
    # repeat training for several randomized datasets
    for i in range(nD):
        # theta = 10.0
        w = np.zeros(len(data[0][0][1]))  # initialize the weights as zero
        # w[len(w)-1] = theta

        X = data[i][0]
        Y = data[i][1]

        # implement sequential perceptron training by cyclic presentation of the P examples
        for epochs in range(n_max):  # stop when n > n_max = 100
            done = True
            for j in range(P):  # this loop runs the P examples
                E = np.dot(X[j], w)*Y[j]  # the local potential
                # we only modify the weights when E<=0. Otherwise they stay the same
                c = 0
                if E <= c:
                    w = w + (1/N)*X[j]*Y[j]
                    done = False
            if done == True:
                success += 1
                break
            
    return success


def test_runs(n_max, nD, N, alpha, ax):
    success_list = []
    for a in alpha:
        success_list.append(experiment(N, n_max, nD, a))

    norm_success = np.divide(np.array(success_list), nD)
    # print('alpha: {}'.format(alpha))
    # print('success_list: {}'.format(success_list))
    # print('norm_success: {}'.format(norm_success))

    ax.plot(alpha, norm_success, label='N: {}'.format(N))


if __name__ == "__main__":

    # Run your code in order to study Perceptron training at least for the following parameter settings:
    # N = 20, P = αN with α = 0.75, 1.0, 1.25, . . . 3.0, nD =50, n_max =100
    n_max = 100
    nD = 50
    N_array = [5, 10, 15, 20, 50, 100]

    # np.random.seed(0)  # To make reproducible sets (if needed)

    # determine the value of the fraction of successful runs as a function of alpha=P/N
    alpha = np.arange(0.75, 3.25, 0.25)

    fig = plt.figure()
    ax = plt.subplot(111)

    for N in progressbar.progressbar(N_array):
        test_runs(n_max, nD, N, alpha, ax)

    plt.xlabel(r'$\alpha = P/N$')
    plt.ylabel(r'$Q_{l.s.}$')
    ax.legend()
    fig.savefig('graphs/Q-alpha-graph.png')
    plt.show()

