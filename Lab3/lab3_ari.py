import numpy as np
import matplotlib as plt
import pandas as pd
from numpy.linalg import norm

dataPath = './Data/'

data3_xi = np.array(pd.read_csv(dataPath+'data3_xi.csv', header=None))
data3_tau = np.array(pd.read_csv(dataPath+'data3_tau.csv', header=None))

P = 100  
learning_rate = 0.05
t_max = 30 # we have to check which number we put here

# initialize weights as independent random vectors (we initialize the weights as unit vectors)
mu, variance = 0, 1
sigma = np.sqrt(variance)

w1 = np.random.normal(mu, sigma, 100)
w2 = np.random.normal(mu, sigma, 100)

mag_w1 = np.linalg.norm(w1)
mag_w2 = np.linalg.norm(w2)

w1_norm = w1/mag_w1
w2_norm = w2/mag_w2

for t in range(t_max): #these are the epochs
    for i in range(P): 
        # perform the training here
        # we have to calculate the derivative of E in order to minimize the cost function
        # SHOULD i BE PICKED AT RANDOM HERE?
        sigma_xi = (np.tanh(np.dot(w1_norm, data3_xi[i]))+np.tanh(np.dot(w2_norm, data3_xi[i])))
        e_nu = (sigma_xi - data3_tau[i])**2/2
        
        #recalculate the weights
        w1_norm = w1_norm - learning_rate * gradient(w1_norm) * e_nu
        w2_norm= w2_norm - learning_rate * gradient(w2_norm) * e_nu
        
        
def gradient():

