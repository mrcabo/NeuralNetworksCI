import numpy as np
import matplotlib.pyplot as plt

# Run your code in order to study Perceptron training at least for the following parameter settings:
# N = 20, P = αN with α = 0.75, 1.0, 1.25, . . . 3.0, nD =50, n_max =100

# I put it in a function so that we can change the values for alpha easily
def experiment(N, n_max, alpha):
    P = int(round(alpha*N))  # number of data points (rounded to closest int)
    n_D = 50 # number of independently generated sets

    mu, variance = 0, 1
    sigma = np.sqrt(variance)
    xi_mu, S_mu = np.zeros([P, N]), np.zeros(P, int)  # TODO: should we add the threshold theta?
    
    data=[]
    # generate 3 datasets
    for i in range(3):
        for i in range(P):
            labels = np.random.binomial(1, 0.5, 1)  # number of trials, probability of each trial
            # result of flipping a coin, tested N times.
            labels[labels < 1] = -1
            S_mu[i] = labels
            xi_mu[i] = np.random.normal(mu, sigma, N)
        data.append([xi_mu,S_mu])
        
    success = perceptron(data, N,P,n_max)
    return success

def perceptron(data,N,P,n_max):
    
    w = np.zeros(len(data[0][0])) # initialize the weights as zero
   
    success=0
    
    # repeat training for several randomized datasets
    for i in range(3):
        X=data[i][0] 
        Y=data[i][1]

        epochs = 0 #count the amount of epochs

        # implement sequential perceptron training by cyclic presentation of the P examples
        # stop when n > n_max = 100
        while epochs <= n_max: 

            # this loop runs the P examples
            for i in range(P):
               
                # I GET AN ERROR HERE BC X[i] HAS 20 ELEMENTS (N) and w has 15 (P). 
                # SO I PUT X INSTEAD BUT STILL WON'T WORK BC OF NESTED LISTS
                # BEFORE IT WORKED BC WE HAD N=P
                E=np.dot(X[i], w)*Y[i] # the local potential 

                # we only modify the weights when E<=0. Otherwise they stay the same
                if E <= 0:
                    w = w + (1/N)*X[i]*Y[i]  
                    
            # is this what they mean by saying that it should stop if all E > 0  ?
            if all(np.dot(elem, w)*l > 0 for elem in X for l in Y):
                success+=1 # we need the ammount of succesful runs to determine Q_ls
                break
            
            epochs+=1
    return success


# determine the value of the fraction of successful runs as a function of alpha=P/N 
alpha=np.arange(0.75,3.25,0.25)
success_list=[]
for a in alpha:
    success_list.append(experiment(20, 100, a))

print(success_list)

# calculate the probability 