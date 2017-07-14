import numpy as np


def cost(param):
    cost = 0
    #run trial, add to cost
    return cost

def probability():
    pass

def add_noise(param, num_params, variance):
    noise = np.random.normal(0,np.sqrt(variance),num_params)
    for k in range(num_params):
        param[k] += noise[k]
        

K = 10
num_params = 12

#cost-to-go = []
prob = []
param = np.zeroes((K, num_params))

#for k in range (K):
#    cost-to-go.append(Cost(param[k:])) #necessary?
    
    
for k in range (K):
    prob.append(probability(cost(param[k:])))
    
    