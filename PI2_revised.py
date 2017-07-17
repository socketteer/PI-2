import ctypes
import numpy as np

from numpy import genfromtxt

# Load cassie library
libcassie = ctypes.CDLL('libcassie.so')
c_double_p = ctypes.POINTER(ctypes.c_double)
libcassie.cassie_init.argtypes = []
libcassie.cassie_init.restype = ctypes.c_void_p
libcassie.cassie_duplicate.argtypes = [ctypes.c_void_p]
libcassie.cassie_duplicate.restype = ctypes.c_void_p
libcassie.cassie_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcassie.cassie_copy.restype = None
libcassie.cassie_free.argtypes = [ctypes.c_void_p]
libcassie.cassie_free.restype = None

libcassie.cassie_step1.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_step1.restype = None
libcassie.cassie_step2.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_step2.restype = None

libcassie.cassie_vis_init.argtypes = []
libcassie.cassie_vis_init.restype = ctypes.c_void_p
libcassie.cassie_vis_free.argtypes = [ctypes.c_void_p]
libcassie.cassie_vis_free.restype = None
libcassie.cassie_vis_draw.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcassie.cassie_vis_draw.restype = ctypes.c_bool

libcassie.cassie_time.argtypes = [ctypes.c_void_p]
libcassie.cassie_time.restype = ctypes.c_double
libcassie.cassie_pos.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_pos.restype = None
libcassie.cassie_vel.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_vel.restype = None
 
    
def cost():
    #calculate cost using state information
    return 0 #returns scalar cost-to-go for timestep   
    
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
        
    prob /= prob_sum #prob is 1D array here   
    
def probability(sensitivity):
    return np.exp(-1/sensitivity * cost()) #returns scalar
    
def generate_noise(num_traj, num_params, num_activation, variance):
    return np.random.normal(0,np.sqrt(variance),(num_traj, num_params, num_activation)) #returns 3D array

def avg_over_trials(num_traj, num_params, num_activation, prob, noise, M):
    avg = np.zeros((num_params, num_activation))
    #for each timestep i
    for p in range(num_params): #for each parameter
        for k in range(num_traj): #loop through trajectories
            #incrementing average by noise of parameter weighted by probability of that trajectory
            avg[p,:] += prob[k] * M[k] * noise[k,p,:]        
    return avg #returns 2D array   

def avg_over_timesteps(dmp, num_params, num_activation, timesteps, timestep_param_update):
    avg = np.zeros((num_params, num_activation))
    denominator = 0 #to scale avg
    for i in range(timesteps - 1): #for each timestep
        for j in range(num_activation):
            #param update for basis j  for all dmp dimensions incremented by timestep param update weighted by corresponding activation  
            avg[:,j] += (timesteps - i) * dmp.get_activations(activation = j, time = i) * \
                timestep_param_update[i,:,j]     
            denominator += dmp.get_activations(activation = j, time = i) * (timesteps - i)
                                                               
    return avg / denominator #returns 2D array

def update_params(dmp, timesteps, num_traj, num_params, num_activation, sensitivity = 1, control_ratio = 0.7):
    #creating buffers
    prob = np.zeros((num_traj, timesteps))
    #base_trajectory = np.zeros((num_params, num_activation)) 
    noise = generate_noise(num_traj, num_params, num_activations, variance)
    timestep_param_update = np.zeros((timesteps, num_params, num_activation))
    param_update = np.zeros((num_params, num_activation))
    R = np.identity(num_params) * control_ratio #??? how, if at all, is this affected by number of activations?
    M = np.zeros((num_traj, timesteps))
    
    base_trajectory = dmp.getweights() 
    
    getting probability and M values for each trajectory
    for k in range (num_traj):
            rollout(start_params = base_trajectory + noise[k,:,:], timesteps, prob[k:], sensitivity, M[k:], R)

    #averaging over trials, getting param update for each timestep    
    for i in range (timesteps):
        timestep_param_update[i,:,:] = avg_over_trials(num_traj, num_params, num_activation, \
            prob[:,i], noise, M[:i])
    
    #averaging over timesteps, getting final param update    
    param_update = avg_over_timesteps(dmp, num_params, num_activation, timesteps, timestep_param_update) 

    return base_trajectory + param_update #sum of 2D arrays

    

timesteps = 30
num_traj = 10
num_params = 12 
num_activation = 10
sensitivity = 1
control_ratio = 0.7

update_params(dmp, timesteps, num_traj, num_params, num_activation)

#

