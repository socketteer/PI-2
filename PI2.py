import numpy as np

#make better version with matrices later

def cost(trajectory):
    cost = 1
    #run trial, add to cost
    return cost

def probability(trajectories, num_params, sensitivity):
    prob = np.zeros(num_params)
    prob_sum = 0
    #trajectories with lower costs have higher probabilities
    for k in range (num_params):
        prob[k] = (np.exp(-1/sensitivity * cost(trajectories[k])))
        prob_sum += prob[k]
    return prob / prob_sum

def generate_noise(num_traj, num_params, variance):
    return np.random.normal(0,np.sqrt(variance),(num_traj, num_params))
                
def avg_over_trials(num_traj, num_params, prob, noise):
    avg = np.zeros(num_params)
    for k in range(num_traj):
        avg += prob[k]*noise[k:]
    return avg    

num_traj = 10
num_params = 12
sensitivity = 1
prob = np.zeros(num_params)
base_trajectory = np.zeros(num_params) 
variance = 1
noise = generate_noise(num_traj, num_params, variance)

for k in range (num_traj):
    #find probabilities of trajectories with gaussian noise
    prob[k] = (probability(base_trajectory + noise[k:], num_params, sensitivity))
    
