import ctypes
import numpy as np

from numpy import genfromtxt
from dmp import DMP, RhythmicDMP

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
 
#TODO elitism
#TODO Pi-2 CMA

class PI2():
    def __init__(self, cassie, dmp, params):
        self.dmp = dmp
        self.params = params
        #TODO self.cassie
    
    def reset_cassie(self):
        #TODO reset cassie state 
        pass
    

    def update_params(self, timesteps, num_traj, variance = 1, sensitivity = 1, control_ratio = 0.7):
        '''        
        timesteps: int, number of timesteps for rollouts.
        num_traj: int, number of noisy trajectories to create rollouts for. 
        variance: array(num_params) of floats or float. variance of noise. If float, 
            all dimensions receive noise with the same variance
        sensitivity: float. determines probability's sensitivity to cost
        control_ratio: array(num_params) of floats or float. If float, all dimensions have 
            the same control cost
            
            
        PI^2 Algorithm: 
            Creates num-traj rollouts from same start state with stochastic DMP parameters.
            Datermines cost of each trial, probability of each trial, averages over trials,
            averages over timesteps, and then updates parameter with weighted average. 

        Returns array((params, num_activation)) of updated DMP weights

            
        '''    
        self.num_params = self.params['dims']
        self.num_activation = self.params['basis_dims'] #basis_dims is placeholder
        
        #If variance or control ratio is a constant, it becomes an array filled with that constant
        #If variance or control ratio is an array, it remains that array
        ones_array = np.ones(self.num_params)
        variance = variance * ones_array
        control_ratio = control_ratio * ones_array
        
        #creating buffers    
        prob = np.zeros((num_traj, timesteps))
        timestep_param_update = np.zeros((timesteps, self.num_params, self.num_activation))
        param_update = np.zeros((self.num_params, self.num_activation))
        M = np.zeros((num_traj, timesteps))
        
        R = np.identity(self.num_params) * control_ratio
        
        noise = generate_noise(num_traj, variance)
        
        base_trajectory = self.dmp.weights #weights is placeholder
        
        #getting probability and M values for K noisy trajectories
        for k in range (num_traj):
            self.rollout(base_trajectory + noise[k,:,:], timesteps, prob[k,:], sensitivity, M[k,:], R)
        
        #averaging over trials, getting param update for each timestep    
        for i in range (timesteps):
            timestep_param_update[i,:,:] = avg_over_trajectories(num_traj, prob[:,i], noise, M[:,i])
        
        #averaging over timesteps, getting final param update    
        param_update = avg_over_timesteps(timesteps, timestep_param_update) 
    
        return base_trajectory + param_update 
    
    def generate_noise(num_traj, variance):
        '''returns float array((num_traj, num_params, num_activation))'''
        #TODO different variances for different DMPs/parameters
        return np.random.normal(0,np.sqrt(variance),(num_traj, self.num_params, self.num_activation)) 
        

    def rollout(self, params, timesteps, prob, sensitivity, M, R):
        '''
        params: array(num_params) of floats. initial weights. 
        prob: array(timesteps) of floats
        M: array(timesteps) of floats
        
        Resets mujoco and dmp state. Runs a rollout of specified trajectory and determines 
        cost-to-go at each timestep.
        The purpose of this method is to compute prob and M for each timestep. prob and M are one
        dimensional slices of the 2 dimensional prob and M defined in update_params. 
        This method is called for each trajectory.'''
        
        cost = np.zeros(timesteps)
        prob_sum = 0
        init_pos = np.zeros(35)
        init_vel = np.zeros(32)
        init_acc = np.zeros(32)
        
        #resetting cassie and copying initial state
        self.reset_cassie()
        libcassie.copy_data(self.cassie, init_pos, init_vel, init_acc)
        
        #resetting dmp and setting weights in params dict
        self.dmp.reset()
        self.params['weights'] = params
 
        for i in range(timesteps):
            cassie_output[8:] = self.dmp.step(self.params) 
            #TODO possible issue: mismatch between dt and step size -- this will work for a dt of 0.0005 
            libcassie.cassie_step2(c, cassie_output.ctypes.data_as(c_double_p))
            libcassie.cassie_step1(c, cassie_input.ctypes.data_as(c_double_p))        
            
            #computing instantaneous cost
            cost[i] = cost()
        
            #self.params['goals'] is a 1D array
            M[i] = np.transpose(R)*self.params['goals']*np.transpose(self.params['goals']) \
                / (np.transpose(self.self.params['goals'])*np.transpose(R)*self.params['goals'])

        #appending terminal cost to each instantaneous cost
        cost += terminal_cost(init_pos, init_vel, init_acc)
     
        #computing probability for each timestep
        for i in range(timesteps):        
            prob[i] = probability(sensitivity, cost[i])
            prob_sum += prob[i]
        
        #prob is 1D array 
        prob /= prob_sum   
        
     def cost():
        '''Computes cost-to-go using state information. Returns a scalar.'''
        cost = 0
        #quadratic distance between center of mass and center of pressure
        stability_cost_x = (left.foot_pos[x] + right.foot_pos[x])**2
        
        #TODO height deviation with forgiveness region
        
        #TODO linear velocity deviation penalty with forgiveness region
        
        return 0 #returns scalar cost-to-go for timestep      
        
     def terminal_cost(init_pos, init_vel, init_acc):
         '''Computes error between initial and final mujoco states. Returns a scalar cost.'''
         #TODO seperate pos, vel, acc
         return (init_state - final_state)**2 #should this be squared or abs?  

     def probability(sensitivity, cost):
         '''Returns a float probability. Probability decays exponentially relative to cost'''
         return np.exp(-1/sensitivity * cost)

     def avg_over_trajectories(num_traj, prob, noise, M):
         '''
         Computes parameter update for each timestep by averaging noise across trajectories weighted
         by probability. This function is called for each timestep.
         
         returns float array((num_params, num_activation))
         '''
         avg = np.zeros((self.num_params, self.num_activation))
         
         for p in range(self.num_params): 
             for k in range(num_traj): 
                 #incrementing average by noise of parameter weighted by probability of that trajectory
                 avg[p,:] += prob[k] * M[k] * noise[k,p,:]        
         return avg    

    def avg_over_timesteps(timesteps, timestep_param_update):
        '''timestep_param_update: float array((timesteps, num_params, num_activation))
        
        This function takes in the parameter update for each timestep and averages across timesteps,
        each paramater update eighted by the number of steps left in the trajectory and the 
        activation of the corresponding basis function in the DMP at that timestep
        
        returns float array((num_params, num_activation))                 
        '''
        avg = np.zeros((self.num_params, self.num_activation))
        denominator = 0 #to scale avg
        for i in range(timesteps - 1): #for each timestep
            for j in range(self.num_activation):
                #param update for basis j across all dmp dimensions incremented by timestep param update 
                #weighted by corresponding activation and steps left in trajectory  
                #get_activation is a placeholder
                avg[:,j] += (timesteps - i) * dmp.get_activations(activation = j, time = i) * \
                    timestep_param_update[i,:,j]     
                denominator += dmp.get_activations(activation = j, time = i) * (timesteps - i)
                                                                   
        return avg / denominator 