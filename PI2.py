import numpy as np

#make better version with matrices later

def cost(trajectory, timesteps):
    cost = np.zeros(timesteps)
    for i in range(timesteps):
        #rollout
        cost[i] = 0
    #run trial, add to cost
    return cost #returns array of costs over timesteps

def probability(trajectories, num_traj, sensitivity, timesteps):
    prob = np.zeros(num_params)
    prob_sum = 0
    #trajectories with lower costs have higher probabilities
    for k in range (num_traj):
        prob[k] = (np.exp(-1/sensitivity * cost(trajectories[k], timesteps)))
        prob_sum += prob[k]
    return prob / prob_sum

def generate_noise(num_traj, num_params, variance):
    return np.random.normal(0,np.sqrt(variance),(num_traj, num_params))
                
def avg_over_trials(num_traj, num_params, prob, noise):
    avg = np.zeros(num_params)
    for k in range(num_traj):
        avg += prob[k]*noise[k:]
    return avg    

def avg_over_timesteps(num_traj, num_params, timesteps):
    pass

timesteps = 30
num_traj = 10
num_params = 12
sensitivity = 1
prob = np.zeros((timesteps,num_params))
base_trajectory = np.zeros(num_params) 
variance = 1
noise = generate_noise(num_traj, num_params, variance)
timestep_param_update = np.zeros((timesteps, num_params))
param_update = np.zeros(num_params)

for k in range (num_traj):
    #find probabilities of trajectories with gaussian noise
    prob[:k] = (probability(base_trajectory + noise[k:], num_params, sensitivity, timesteps))
 
for i in range (timesteps):
    for k in range (num_traj):
        #calculating parameter update for each timestep
        timestep_param_update[i:] = avg_over_trials(num_traj, num_params, prob[i,k], noise) 

