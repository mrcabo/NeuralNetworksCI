import numpy as np
import matplotlib.pyplot as plt


# I put it in a function so that we can change the values for alpha easily
def experiment(N, n_max, nD, alpha):
    P = int(round(alpha*N))  # number of data points (rounded to closest int)
    mu, variance = 0, 1
    sigma = np.sqrt(variance)
    xi_mu, S_mu = np.zeros([P, N]), np.zeros(P, int)  # TODO: should we add the threshold theta?
    
    data=[]
    # generate nD datasets
    for i in range(nD):
        for i in range(P):
            labels = np.random.binomial(1, 0.5, 1)  # number of trials, probability of each trial
            # result of flipping a coin, tested N times.
            labels[labels < 1] = -1
            S_mu[i] = labels
            xi_mu[i] = np.random.normal(mu, sigma, N)
        data.append([xi_mu,S_mu])
        
    success = perceptron(data, N, P, n_max)
    return success


def perceptron(data, N, P, n_max):
    
    w = np.zeros(len(data[0][0][1]))  # initialize the weights as zero
    success=0
    # repeat training for several randomized datasets
    for i in range(nD):
        X = data[i][0]
        Y = data[i][1]

        # implement sequential perceptron training by cyclic presentation of the P examples
        for epochs in range(n_max):  # stop when n > n_max = 100
            done = True
            for j in range(P):  # this loop runs the P examples
                E = np.dot(X[j], w)*Y[j]  # the local potential
                # we only modify the weights when E<=0. Otherwise they stay the same
                if E <= 0:
                    w = w + (1/N)*X[j]*Y[j]
                    done = False
            if done:
                success += 1
                break
            
    return success


if __name__ == "__main__":

    # Run your code in order to study Perceptron training at least for the following parameter settings:
    # N = 20, P = αN with α = 0.75, 1.0, 1.25, . . . 3.0, nD =50, n_max =100
    n_max = 100
    nD = 50
    N = 20

    #  As mentioned in assignment, set parameters as large as possible (within reason...)
    n_max *= 5
    nD *= 6
    N *= 5

    # determine the value of the fraction of successful runs as a function of alpha=P/N
    alpha=np.arange(0.75, 3.25, 0.25)

    success_list=[]
    for a in alpha:
        success_list.append(experiment(N, n_max, nD, a))

    norm_success = np.divide(success_list, nD)

    print('alpha: {}'.format(alpha))
    print('success_list: {}'.format(success_list))
    print('norm_success: {}'.format(norm_success))

    plt.plot(alpha, norm_success)
    plt.xlabel(r'$\alpha = P/N$')
    plt.ylabel(r'$Q_{l.s.}$')
    plt.savefig('Q-alpha-graph.png')
    plt.show()
