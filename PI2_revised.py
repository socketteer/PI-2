import numpy as np
 
    
def cost():
    #calculate cost using state information
    return 0    
    
def rollout(dmp, start_params, num_params, num_activation, timesteps, prob, sensitivity, M, R):
    #cost = np.zeros(timesteps)
    prob_sum = 0
    #reset DMP here
    #copy cassie
    #set weights in params
    
    for i in range(timesteps):
        cassie_output[8:] = dmp.step(params) 
        libcassie.cassie_step2(c, cassie_output.ctypes.data_as(c_double_p))
        libcassie.cassie_step1(c, cassie_input.ctypes.data_as(c_double_p))        
        
        prob[i] = probability(sensitivity)
        prob_sum += prob[i]
        
        #computing M, calling goals function from DMP 
        M[i] = np.transpose(R)*dmp.goals()*np.transpose(dmp.goals()) \
            / (np.transpose(dmp.goals())*np.transpose(R)*dmp.goals())
        
    prob /= prob_sum     
    
def probability(sensitivity):
    return np.exp(-1/sensitivity * cost())
    
def generate_noise(num_traj, num_params, num_activation, variance):
    return np.random.normal(0,np.sqrt(variance),(num_traj, num_params, num_activation)

def avg_over_trials(num_traj, num_params, num_activation, prob, noise, M):
    avg = np.zeros(num_params, num_activation)
    #for each timestep i
    for p in range(num_params): #for each parameter
        for k in range(num_traj): #loop through trajectories
            #incrementing average by noise of parameter weighted by probability of that trajectory
            avg[p,:] += prob[k] * M[k] * noise[k,p,:]
        
    return avg    

def avg_over_timesteps(num_params, timesteps):
    pass


timesteps = 30
num_traj = 10
num_params = 12 
num_activation = 10
sensitivity = 1
prob = np.zeros((num_traj, timesteps))
base_trajectory = np.zeros((num_params, num_activation)) 
noise = generate_noise(num_traj, num_params, variance)
timestep_param_update = np.zeros((timesteps, num_params, num_activation))
param_update = np.zeros((num_params, num_activation))
control_ratio = 0.7
R = np.identity(num_params) * control_ratio #??? how, if at all, is this affected by number of activations?
M = np.zeros((num_traj, timesteps))

base_trajectory = dmp.getweights() 

for k in range (num_traj):
    rollout(start_params = base_trajectory + noise[k,:,:], timesteps, prob[k:], sensitivity, M[k:], R)
    
for i in range (timesteps):
    timestep_param_update[i,:,:] = avg_over_trials(num_traj, num_params, prob[:,i], noise, M[:i])
    
param_update = avg_over_timesteps(num_params, timesteps) 

    